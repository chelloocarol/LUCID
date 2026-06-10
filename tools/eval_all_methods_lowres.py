"""
Compute PSNR/SSIM/MAE/Vis for ALL methods at a common low resolution (448x256).
This enables fair comparison including AdaIR (which was inferred at 448x256 due to CPU limits).

Method outputs are resized from their native resolution to the target resolution.
Targets (RIDCP pseudo-labels) are also resized consistently.

Usage:
    python tools/eval_all_methods_lowres.py \
        --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
        --eval_size 448 256 \
        --output_json experiment/eval/comparison_lowres_448x256.json
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFER_ROOT = os.path.join(ROOT, "experiment", "infer_test")

# Methods and their output directories (relative to INFER_ROOT)
# AdaIR is already at 448x256; others will be downscaled from native
METHODS = {
    "input": "input",
    "DCP": "dcp",
    "CLAHE": "clahe",
    "Retinex": "retinex",
    "骨干基线(Student)": "student",
    "LUCIDMine": "lucidmine_modal_v2",
    "AdaIR(ICLR2025)": "adair_dehaze",
}


def load_image(path, size_wh=None):
    img = Image.open(path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def compute_vis_component(img_np):
    """Compute raw (un-normalised) Vis sub-metrics for a single image."""
    gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2])
    h, w = gray.shape
    # Contrast (Michelson over 15x15 windows, average)
    from scipy.ndimage import uniform_filter
    local_max = uniform_filter(img_np.max(axis=2), size=15)
    local_min = uniform_filter(img_np.min(axis=2), size=15)
    denom = local_max + local_min
    contrast = np.where(denom > 0, (local_max - local_min) / denom, 0)
    # Entropy
    hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
    # Dark channel
    dark = img_np.min(axis=2)
    dark_ch = float(uniform_filter(dark, size=15).mean())
    # Glare ratio
    max_ch = img_np.max(axis=2)
    glare = float((max_ch > 0.95).mean())
    # Sharpness (Laplacian variance)
    from scipy.ndimage import laplace
    sharp = float(laplace(gray).var())
    return {
        "contrast": float(contrast.mean()),
        "entropy": entropy,
        "dark_channel": dark_ch,
        "glare": glare,
        "sharpness": sharp,
    }


def normalise_vis(records_by_method):
    """Normalise each sub-metric across ALL images from ALL methods."""
    all_records = [r for recs in records_by_method.values() for r in recs]
    keys = ["contrast", "entropy", "dark_channel", "glare", "sharpness"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in all_records]
        stats[k] = (min(vals), max(vals))

    def norm(v, mn, mx):
        return (v - mn) / (mx - mn) if mx > mn else 0.0

    for recs in records_by_method.values():
        for r in recs:
            mn_c, mx_c = stats["contrast"]
            mn_e, mx_e = stats["entropy"]
            mn_d, mx_d = stats["dark_channel"]
            mn_g, mx_g = stats["glare"]
            mn_s, mx_s = stats["sharpness"]
            c_n = norm(r["contrast"], mn_c, mx_c)
            e_n = norm(r["entropy"], mn_e, mx_e)
            d_n = norm(r["dark_channel"], mn_d, mx_d)
            g_n = norm(r["glare"], mn_g, mx_g)
            s_n = norm(r["sharpness"], mn_s, mx_s)
            r["vis"] = 0.25 * c_n + 0.25 * e_n + 0.20 * (1 - d_n) + 0.15 * (1 - g_n) + 0.15 * s_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir",
                        default="/home/user/lucidmine-40-video-dataset/data/test/target")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[448, 256])
    parser.add_argument("--output_json",
                        default=os.path.join(ROOT, "experiment/eval/comparison_lowres_448x256.json"))
    args = parser.parse_args()

    eval_wh = tuple(args.eval_size)
    print(f"Evaluation resolution: {eval_wh[0]}x{eval_wh[1]}")

    # Build target filename map (stem -> path)
    target_map = {os.path.splitext(f)[0]: os.path.join(args.target_dir, f)
                  for f in os.listdir(args.target_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    print(f"Targets: {len(target_map)}")

    results = {}
    records_by_method = {}

    for method_name, subdir in METHODS.items():
        method_dir = os.path.join(INFER_ROOT, subdir)
        if not os.path.isdir(method_dir):
            print(f"SKIP {method_name}: directory not found ({method_dir})")
            continue
        pred_files = sorted([f for f in os.listdir(method_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not pred_files:
            print(f"SKIP {method_name}: no images found")
            continue

        print(f"\n{method_name}: {len(pred_files)} images...")
        psnr_vals, ssim_vals, mae_vals = [], [], []
        vis_records = []

        for fname in tqdm(pred_files, desc=method_name, leave=False):
            stem = os.path.splitext(fname)[0]
            if stem not in target_map:
                continue

            pred = load_image(os.path.join(method_dir, fname), size_wh=eval_wh)
            target = load_image(target_map[stem], size_wh=eval_wh)

            p = psnr_fn(target, pred, data_range=1.0)
            s = ssim_fn(target, pred, data_range=1.0, channel_axis=2)
            m = float(np.mean(np.abs(pred - target)))

            psnr_vals.append(p)
            ssim_vals.append(s)
            mae_vals.append(m)

            vis_raw = compute_vis_component(pred)
            vis_raw["stem"] = stem
            vis_records.append(vis_raw)

        if not psnr_vals:
            print(f"  No matched pairs for {method_name}")
            continue

        results[method_name] = {
            "n": len(psnr_vals),
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "mae": float(np.mean(mae_vals)),
            "psnr_std": float(np.std(psnr_vals)),
        }
        records_by_method[method_name] = vis_records
        print(f"  {method_name}: PSNR={results[method_name]['psnr']:.4f}, "
              f"SSIM={results[method_name]['ssim']:.4f}, MAE={results[method_name]['mae']:.4f}")

    # Normalise Vis across all methods
    print("\nNormalising Vis scores across all methods...")
    normalise_vis(records_by_method)
    for method_name, vis_records in records_by_method.items():
        vis_scores = [r["vis"] for r in vis_records]
        results[method_name]["vis"] = float(np.mean(vis_scores))
        print(f"  {method_name}: Vis={results[method_name]['vis']:.6f}")

    output = {
        "eval_resolution": f"{eval_wh[0]}x{eval_wh[1]}",
        "n_per_method": "up to 152",
        "note": (
            f"All methods evaluated at {eval_wh[0]}x{eval_wh[1]} for fair comparison. "
            "Outputs resized from native resolution. Targets resized from 1920x1088. "
            "These numbers are NOT directly comparable to full-resolution numbers in other tables."
        ),
        "methods": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_json}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'PSNR':>8} {'SSIM':>8} {'MAE':>8} {'Vis':>8}")
    print(f"{'-'*70}")
    for mname, r in results.items():
        print(f"{mname:<25} {r['psnr']:>8.4f} {r['ssim']:>8.4f} {r['mae']:>8.4f} {r.get('vis', float('nan')):>8.4f}")
    print(f"{'='*70}")
    print(f"Resolution: {eval_wh[0]}x{eval_wh[1]} (all targets resized)")


if __name__ == "__main__":
    main()
