"""
Run AdaIR dehazing inference on a directory of images.
Uses adair-single-dehaze.ckpt (ICLR 2025 baseline).

For CPU feasibility, images are resized to --eval_size before inference.
Targets are also resized to the same size for fair metric computation.

Usage:
    python tools/infer_adair.py \
        --input_dir /home/user/lucidmine-40-video-dataset/data/test/input \
        --output_dir experiment/infer_test/adair_dehaze \
        --adair_dir /home/user/AdaIR \
        --ckpt /home/user/AdaIR/ckpt/models/adair-single-dehaze.ckpt \
        --eval_size 448 256
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adair_dir", default="/home/user/AdaIR")
    p.add_argument("--ckpt", default="/home/user/AdaIR/ckpt/models/adair-single-dehaze.ckpt")
    p.add_argument("--eval_size", type=int, nargs=2, default=[448, 256],
                   help="W H to resize input for inference (divisible by 32)")
    p.add_argument("--num_threads", type=int, default=4)
    return p.parse_args()


def load_model(adair_dir, ckpt_path):
    sys.path.insert(0, adair_dir)
    import lightning.pytorch as pl
    from net.model import AdaIR
    import torch.nn as nn

    class AdaIRModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = AdaIR(decoder=True)
            self.loss_fn = nn.L1Loss()
        def forward(self, x):
            return self.net(x)

    device = torch.device("cpu")
    print(f"Loading checkpoint: {ckpt_path}")
    model = AdaIRModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params, CPU mode)")
    return model, device


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.output_dir, exist_ok=True)

    model, device = load_model(args.adair_dir, args.ckpt)
    to_tensor = ToTensor()

    eval_w, eval_h = args.eval_size
    print(f"Inference resolution: {eval_w}x{eval_h}")

    img_files = sorted([f for f in os.listdir(args.input_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(img_files)} images")

    for fname in tqdm(img_files, desc="AdaIR inference"):
        out_path = os.path.join(args.output_dir, fname)
        if os.path.exists(out_path):
            continue

        img = Image.open(os.path.join(args.input_dir, fname)).convert("RGB")
        # resize to eval_size for inference (keeping aspect close to eval_w x eval_h)
        img_resized = img.resize((eval_w, eval_h), Image.LANCZOS)
        tensor = to_tensor(img_resized).unsqueeze(0)

        with torch.no_grad():
            restored = model(tensor.to(device)).cpu()

        restored = restored.clamp(0, 1).squeeze(0)
        out_arr = (restored.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # save at eval_size (targets will be resized to same size for metrics)
        Image.fromarray(out_arr).save(out_path)

    count = len(os.listdir(args.output_dir))
    print(f"Done. {count} images saved to {args.output_dir}")
    print(f"NOTE: outputs are {eval_w}x{eval_h}; compute metrics against targets resized to same size")


if __name__ == "__main__":
    main()
