"""
Compute PSNR/SSIM/MAE/Vis for AdaIR inference outputs at 448x256.
Targets (RIDCP pseudo-labels) are resized to the same resolution for fair comparison.

Usage:
    python tools/eval_adair_metrics.py \
        --pred_dir experiment/infer_test/adair_dehaze \
        --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
        --eval_size 448 256 \
        --output_json experiment/eval/adair_eval.json
"""

import argparse
import json
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--target_dir", required=True)
    p.add_argument("--eval_size", type=int, nargs=2, default=[448, 256])
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def load_image(path, size_wh=None):
    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def compute_masked_mae(pred, target, mask_threshold=0.02):
    """Compute MAE only on non-trivial pixels (exclude near-black border)."""
    mask = target.max(axis=2) > mask_threshold
    if mask.sum() == 0:
        return float(np.mean(np.abs(pred - target)))
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def main():
    args = parse_args()
    eval_wh = tuple(args.eval_size)  # (W, H)

    pred_files = sorted([f for f in os.listdir(args.pred_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    target_files = {os.path.splitext(f)[0]: f
                    for f in os.listdir(args.target_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

    print(f"Pred: {len(pred_files)}, Targets available: {len(target_files)}")

    psnr_vals, ssim_vals, mae_vals = [], [], []
    missing = 0

    for fname in tqdm(pred_files, desc="Evaluating AdaIR"):
        stem = os.path.splitext(fname)[0]
        if stem not in target_files:
            missing += 1
            continue

        pred = load_image(os.path.join(args.pred_dir, fname))
        # pred is already at eval_size; target needs to be resized
        target = load_image(os.path.join(args.target_dir, target_files[stem]), size_wh=eval_wh)

        p = psnr(target, pred, data_range=1.0)
        s = ssim(target, pred, data_range=1.0, channel_axis=2)
        m = compute_masked_mae(pred, target)

        psnr_vals.append(p)
        ssim_vals.append(s)
        mae_vals.append(m)

    n = len(psnr_vals)
    result = {
        "method": "AdaIR-single-dehaze",
        "eval_resolution": f"{eval_wh[0]}x{eval_wh[1]}",
        "n": n,
        "missing": missing,
        "full_psnr": float(np.mean(psnr_vals)),
        "full_ssim": float(np.mean(ssim_vals)),
        "masked_l1": float(np.mean(mae_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "mae_std": float(np.std(mae_vals)),
        "note": (
            "Evaluated at 448x256 (CPU inference limit). Targets resized from 1920x1088. "
            "Not directly comparable to other methods evaluated at full resolution. "
            "AdaIR trained on standard dehazing benchmarks (SOTS), not coal mine domain."
        )
    }

    print(f"\n{'='*60}")
    print(f"AdaIR Evaluation (n={n}, {eval_wh[0]}x{eval_wh[1]})")
    print(f"  PSNR:  {result['full_psnr']:.4f} dB ± {result['psnr_std']:.4f}")
    print(f"  SSIM:  {result['full_ssim']:.4f} ± {result['ssim_std']:.4f}")
    print(f"  MAE:   {result['masked_l1']:.4f} ± {result['mae_std']:.4f}")
    print(f"  NOTE: Results at {eval_wh[0]}x{eval_wh[1]} — see note in JSON")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
