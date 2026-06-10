"""
LUCIDMine supervised fine-tuning script.

Two-stage training as described in the paper:
  Stage 1 (warmup, --warmup_epochs):  freeze backbone, train VCA+GARC+gates only
  Stage 2 (adapt,  --adapt_epochs):   unfreeze bottleneck+decoder, joint end-to-end

Usage (local CPU smoke test):
  python train_lucidmine.py \
    --manifest data/mine_manifest.csv \
    --init_ckpt weights/Student.pth \
    --exp_name smoke_lucidmine \
    --warmup_epochs 1 --adapt_epochs 1 \
    --batch_size 2 --crop_size 128 --num_workers 0

Usage (full, GPU):
  python train_lucidmine.py \
    --manifest data/mine_manifest.csv \
    --init_ckpt weights/Student.pth \
    --exp_name lucidmine_full \
    --warmup_epochs 20 --adapt_epochs 80 \
    --batch_size 8 --crop_size 256 --num_workers 4
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data import MineManifestDataset
from loss import SSIM, MaskedL1Loss, masked_psnr
from metric import ssim, psnr
from model import LUCIDMine


# ---------- Loss functions ----------

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        gx_pred   = nn.functional.conv2d(pred,   self.kx, padding=1, groups=3)
        gy_pred   = nn.functional.conv2d(pred,   self.ky, padding=1, groups=3)
        gx_target = nn.functional.conv2d(target, self.kx, padding=1, groups=3)
        gy_target = nn.functional.conv2d(target, self.ky, padding=1, groups=3)
        return (gx_pred - gx_target).abs().mean() + (gy_pred - gy_target).abs().mean()


class GlareRegLoss(nn.Module):
    """Penalise positive over-restoration in high-glare regions."""

    def forward(self, restored, backbone_out, glare_map):
        over_correction = torch.relu(restored - backbone_out)
        return (glare_map * over_correction).mean()


def build_mine_priors(model, x):
    """Extract full-resolution priors (used for glare reg loss)."""
    return model.mine_prior(x, x.shape[2:])


# ---------- Param groups ----------

def build_param_groups(model, lr_adapter, lr_backbone):
    adapter_params, backbone_params = [], []
    adapter_names = {"mine_prior", "visibility_adapter", "glare_calibrator"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in adapter_names:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": adapter_params, "lr": lr_adapter},
        {"params": backbone_params, "lr": lr_backbone},
    ]


def freeze_backbone(model):
    adapter_names = {"mine_prior", "visibility_adapter", "glare_calibrator"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        param.requires_grad = top in adapter_names


def unfreeze_bottleneck_decoder(model):
    """Unfreeze bottleneck and decoder; keep encoder frozen (as per paper)."""
    encoder_prefixes = (
        "conv_input", "dense0", "conv1",
        "conv2x", "dense1", "conv1.", "conv2.",
        "conv4x", "dense2", "conv2.", "conv3.",
        "conv8x", "dense3", "conv3.", "conv4.",
        "conv16x", "dense4", "conv4.",
        "fusion1", "fusion2", "fusion3", "fusion4",
    )
    for name, param in model.named_parameters():
        # Always allow adapter params
        top = name.split(".")[0]
        if top in {"mine_prior", "visibility_adapter", "glare_calibrator"}:
            param.requires_grad = True
        elif any(name.startswith(p) for p in encoder_prefixes):
            param.requires_grad = False
        else:
            param.requires_grad = True


# ---------- Training ----------

def train_one_epoch(model, loader, optimizer, device, ssim_loss_fn, edge_loss_fn,
                    w_l1, w_ssim, w_edge, w_glare, scaler=None):
    model.train()
    total_loss = total_l1 = total_ssim_l = total_edge = total_glare = 0.0
    n = 0
    for batch in loader:
        haze   = batch["haze"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        ini    = batch["haze"].to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                restored, feats = model(haze)
                restored = restored.clamp(0, 1)
                l1_l = (restored - target).abs().mean()
                edge_l = edge_loss_fn(restored, target)
            # SSIM and its backward must run in fp32 to avoid fp16 variance instability
            restored_f32 = restored.float()
            target_f32   = target.float()
            ssim_l = 1.0 - ssim_loss_fn(restored_f32, target_f32)
            loss = w_l1 * l1_l + w_ssim * ssim_l + w_edge * edge_l
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            restored, feats = model(haze)
            restored = restored.clamp(0, 1)
            l1_l   = (restored - target).abs().mean()
            ssim_l = 1.0 - ssim_loss_fn(restored, target)
            edge_l = edge_loss_fn(restored, target)
            loss   = w_l1 * l1_l + w_ssim * ssim_l + w_edge * edge_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = haze.size(0)
        total_loss  += loss.item() * bs
        total_l1    += l1_l.item() * bs
        total_ssim_l+= ssim_l.item() * bs
        total_edge  += edge_l.item() * bs
        n += bs

    return {
        "loss":  total_loss  / n,
        "l1":    total_l1    / n,
        "ssim":  total_ssim_l / n,
        "edge":  total_edge  / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_vals, ssim_vals, mae_vals = [], [], []
    for batch in loader:
        haze   = batch["haze"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        restored, _ = model(haze)
        restored = restored.clamp(0, 1)
        for i in range(restored.size(0)):
            psnr_vals.append(psnr(restored[i:i+1], target[i:i+1]))
            ssim_vals.append(ssim(restored[i:i+1], target[i:i+1]).item())
            mae_vals.append((restored[i:i+1] - target[i:i+1]).abs().mean().item())
    import numpy as np
    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "mae":  float(np.mean(mae_vals)),
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",       required=True)
    parser.add_argument("--init_ckpt",      default="weights/Student.pth")
    parser.add_argument("--exp_name",       default="lucidmine_run")
    parser.add_argument("--exp_dir",        default="experiment/LUCIDMine")
    parser.add_argument("--warmup_epochs",  type=int, default=20)
    parser.add_argument("--adapt_epochs",   type=int, default=80)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--crop_size",      type=int, default=256)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--lr_adapter",     type=float, default=2e-4)
    parser.add_argument("--lr_backbone",    type=float, default=2e-5)
    parser.add_argument("--end_lr",         type=float, default=1e-6)
    parser.add_argument("--w_l1",           type=float, default=1.0)
    parser.add_argument("--w_ssim",         type=float, default=0.2)
    parser.add_argument("--w_edge",         type=float, default=0.05)
    parser.add_argument("--w_glare",        type=float, default=0.03)
    parser.add_argument("--save_every",     type=int, default=10)
    parser.add_argument("--seed",           type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join(args.exp_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_ds = MineManifestDataset(args.manifest, split="train", train=True,  crop_size=args.crop_size)
    val_ds   = MineManifestDataset(args.manifest, split="val",   train=False, crop_size=args.crop_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type=="cuda", drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type=="cuda")

    # Model
    model = LUCIDMine().to(device)
    ckpt = torch.load(args.init_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded {args.init_ckpt}: missing={len(missing)}, unexpected={len(unexpected)}")

    # Loss
    ssim_fn = SSIM().to(device)
    edge_fn = SobelEdgeLoss().to(device)

    # AMP scaler (GPU only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    total_epochs = args.warmup_epochs + args.adapt_epochs
    history = []
    best_psnr = 0.0

    for epoch in range(1, total_epochs + 1):
        stage = "warmup" if epoch <= args.warmup_epochs else "adapt"

        # Freeze / unfreeze
        if epoch == 1:
            freeze_backbone(model)
            print("Stage 1: backbone frozen, training VCA+GARC only")
        elif epoch == args.warmup_epochs + 1:
            unfreeze_bottleneck_decoder(model)
            print("Stage 2: bottleneck+decoder unfrozen, joint training")

        # Build optimizer fresh when stage changes (or epoch==1)
        if epoch == 1 or epoch == args.warmup_epochs + 1:
            param_groups = build_param_groups(model, args.lr_adapter, args.lr_backbone if stage == "adapt" else 0.0)
            optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-4)
            stage_epochs = args.warmup_epochs if stage == "warmup" else args.adapt_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs, eta_min=args.end_lr)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            ssim_fn, edge_fn,
            args.w_l1, args.w_ssim, args.w_edge, args.w_glare, scaler
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "stage": stage,
            "elapsed_s": round(elapsed, 1),
            **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()},
            **{f"val_{k}":   round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(row)

        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_psnr": best_psnr},
                       os.path.join(save_dir, "best.pth"))

        if epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(save_dir, f"epoch_{epoch:04d}.pth"))

        print(f"[{epoch:3d}/{total_epochs}|{stage}] "
              f"loss={train_metrics['loss']:.4f} "
              f"val_psnr={val_metrics['psnr']:.3f} val_ssim={val_metrics['ssim']:.4f} "
              f"val_mae={val_metrics['mae']:.4f} "
              f"{'★' if is_best else ' '} ({elapsed:.0f}s)")

        # Save history
        with open(os.path.join(save_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Best val PSNR: {best_psnr:.3f} dB")
    torch.save({"model": model.state_dict(), "epoch": total_epochs},
               os.path.join(save_dir, "last.pth"))


if __name__ == "__main__":
    main()
