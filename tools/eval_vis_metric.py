"""
Compute the composite Visibility proxy score (Vis) defined in the paper §3.3:

    Vis = 0.25*C~ + 0.25*E~ + 0.20*(1 - D~) + 0.15*(1 - G~) + 0.15*S~

where each sub-metric is normalised to [0,1] using the range observed across
all images in the provided directories so that results are comparable.
Sub-metrics:
    C~  normalised local-contrast (Michelson over 15×15 window)
    E~  normalised Shannon entropy (8-bit histogram)
    D~  normalised dark-channel mean (DCP-style, 15×15 min-filter)
    G~  normalised glare ratio (fraction of pixels above 0.95 max-channel)
    S~  normalised Laplacian-variance sharpness

Usage:
    # Single-directory (no reference):
    python tools/eval_vis_metric.py \
        --dirs lucidmine=experiment/infer/lucidmine \
               clahe=experiment/infer/clahe \
               input=experiment/infer/input \
        --output_csv experiment/eval/vis_results.csv

    # With paired reference (for full-reference metrics too):
    python tools/eval_vis_metric.py \
        --dirs lucidmine=... input=... \
        --ref_dir /path/to/targets \
        --output_csv experiment/eval/vis_results.csv
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


# ---------- raw sub-metrics (unnormalised) ----------

def michelson_contrast(gray_f32, win=15):
    """Mean local Michelson contrast over non-overlapping win×win patches."""
    h, w = gray_f32.shape
    vals = []
    for y in range(0, h - win + 1, win):
        for x in range(0, w - win + 1, win):
            patch = gray_f32[y:y+win, x:x+win]
            lo, hi = patch.min(), patch.max()
            denom = lo + hi
            vals.append((hi - lo) / denom if denom > 1e-6 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def shannon_entropy(gray_u8):
    hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    return float(-(hist * np.log2(hist)).sum())


def dark_channel_mean(rgb_f32, patch=15):
    min_rgb = rgb_f32.min(axis=2)
    kernel = np.ones((patch, patch), dtype=np.uint8)
    dark = cv2.erode(min_rgb, kernel, borderType=cv2.BORDER_REPLICATE)
    return float(dark.mean())


def glare_ratio(rgb_f32, threshold=0.95):
    """Fraction of pixels where max channel > threshold."""
    return float((rgb_f32.max(axis=2) > threshold).mean())


def laplacian_sharpness(gray_f32):
    """Variance of Laplacian (normalised by max possible for float32)."""
    gray_u8 = (gray_f32 * 255).clip(0, 255).astype(np.uint8)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(lap.var())


def psnr_np(pred, target):
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_np(pred, target):
    """Fast grayscale SSIM."""
    p = pred.mean(axis=2).astype(np.float64)
    t = target.mean(axis=2).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = cv2.GaussianBlur(p, (11, 11), 1.5), cv2.GaussianBlur(t, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    s1 = cv2.GaussianBlur(p*p, (11, 11), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(t*t, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(p*t, (11, 11), 1.5) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float((num / (den + 1e-10)).mean())


# ---------- per-image compute ----------

def compute_raw(image_path):
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray_f32 = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray_u8  = (gray_f32 * 255).clip(0, 255).astype(np.uint8)
    return {
        "contrast":    michelson_contrast(gray_f32),
        "entropy":     shannon_entropy(gray_u8),
        "dark_channel": dark_channel_mean(rgb),
        "glare":       glare_ratio(rgb),
        "sharpness":   laplacian_sharpness(gray_f32),
    }


# ---------- normalise across all images ----------

def normalise_records(records):
    """Min-max normalise each sub-metric across all records in place."""
    keys = ["contrast", "entropy", "dark_channel", "glare", "sharpness"]
    ranges = {}
    for k in keys:
        vals = [r[k] for r in records]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi)

    for r in records:
        for k in keys:
            lo, hi = ranges[k]
            r[f"{k}_norm"] = (r[k] - lo) / (hi - lo) if hi > lo else 0.5
        # Vis composite (paper §3.3)
        C = r["contrast_norm"]
        E = r["entropy_norm"]
        D = r["dark_channel_norm"]
        G = r["glare_norm"]
        S = r["sharpness_norm"]
        r["vis"] = 0.25*C + 0.25*E + 0.20*(1-D) + 0.15*(1-G) + 0.15*S
    return ranges


def list_images(d):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(d, e)))
    return sorted(imgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="name=path pairs, e.g. lucidmine=./out/lucidmine input=./out/input")
    parser.add_argument("--ref_dir",    default="",
                        help="Optional: reference target directory for PSNR/SSIM/MAE")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    # Parse name=path pairs
    method_dirs = {}
    for item in args.dirs:
        name, path = item.split("=", 1)
        method_dirs[name] = path

    # Collect all records
    all_records = []
    for method, directory in method_dirs.items():
        images = list_images(directory)
        if not images:
            print(f"WARNING: no images found in {directory}")
            continue
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            row = {
                "method":   method,
                "stem":     stem,
                "path":     img_path,
            }
            row.update(compute_raw(img_path))
            all_records.append(row)
            print(f"  {method}/{stem}: contrast={row['contrast']:.4f} entropy={row['entropy']:.4f}")

    if not all_records:
        raise ValueError("No images found in any of the specified directories.")

    normalise_records(all_records)

    # Optional reference metrics
    if args.ref_dir:
        ref_map = {}
        for rp in list_images(args.ref_dir):
            ref_map[os.path.splitext(os.path.basename(rp))[0]] = rp
        for r in all_records:
            ref_path = ref_map.get(r["stem"])
            if ref_path:
                pred = np.asarray(Image.open(r["path"]).convert("RGB"))
                tgt  = np.asarray(Image.open(ref_path).convert("RGB"))
                if pred.shape == tgt.shape:
                    r["psnr"] = psnr_np(pred, tgt)
                    r["ssim"] = ssim_np(pred, tgt)
                    r["mae"]  = float(np.abs(pred.astype(np.float64) - tgt.astype(np.float64)).mean() / 255.0)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = [k for k in all_records[0].keys() if k != "path"]
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nWrote {len(all_records)} rows to {args.output_csv}")

    # Per-method summary
    import json
    summary = {}
    for method in method_dirs:
        rows = [r for r in all_records if r["method"] == method]
        if not rows:
            continue
        summary[method] = {
            k: round(float(np.mean([r[k] for r in rows if k in r])), 6)
            for k in ["contrast", "entropy", "dark_channel", "glare", "sharpness", "vis"]
        }
        for k in ["psnr", "ssim", "mae"]:
            vals = [r[k] for r in rows if k in r]
            if vals:
                summary[method][k] = round(float(np.mean(vals)), 6)

    print("\n=== Per-method summary ===")
    for method, m in summary.items():
        print(f"  {method:15s}  vis={m.get('vis', 'n/a'):.4f}  ", end="")
        if "psnr" in m:
            print(f"psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  mae={m['mae']:.4f}", end="")
        print()

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    main()
