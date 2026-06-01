import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def list_images(input_dir):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(images)


def image_entropy(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def laplacian_variance(gray):
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


def dark_channel_mean(rgb, patch=15):
    min_rgb = rgb.min(axis=2)
    kernel = np.ones((patch, patch), dtype=np.uint8)
    dark = cv2.erode(min_rgb, kernel, borderType=cv2.BORDER_REPLICATE)
    return float(dark.mean())


def clipped_ratio(rgb, low=0.02, high=0.98):
    return float(((rgb <= low) | (rgb >= high)).mean())


def compute_metrics(image_path):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return {
        "entropy": image_entropy(gray),
        "laplacian_var": laplacian_variance(gray),
        "dark_channel_mean": dark_channel_mean(rgb),
        "clipped_ratio": clipped_ratio(rgb),
        "mean_luminance": float(gray.mean()),
    }


def summarize(records):
    summary = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if key in {"filename", "path"}:
                continue
            summary[key].append(value)
    return {key: float(np.mean(values)) for key, values in summary.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--reference_dir", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    images = list_images(args.input_dir)
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    records = []
    for image_path in images:
        row = {
            "filename": os.path.basename(image_path),
            "path": image_path,
        }
        row.update(compute_metrics(image_path))
        records.append(row)

    payload = {
        "input_dir": args.input_dir,
        "count": len(records),
        "summary": summarize(records),
        "records": records,
        "note": "These are lightweight proxy metrics for public real-image screening, not official FADE/BIQME replacements.",
    }

    if args.reference_dir:
        ref_images = {os.path.basename(path): path for path in list_images(args.reference_dir)}
        deltas = []
        for row in records:
            ref_path = ref_images.get(row["filename"])
            if not ref_path:
                continue
            ref_metrics = compute_metrics(ref_path)
            deltas.append(
                {
                    "entropy_delta": row["entropy"] - ref_metrics["entropy"],
                    "laplacian_var_delta": row["laplacian_var"] - ref_metrics["laplacian_var"],
                    "dark_channel_mean_delta": row["dark_channel_mean"] - ref_metrics["dark_channel_mean"],
                    "clipped_ratio_delta": row["clipped_ratio"] - ref_metrics["clipped_ratio"],
                    "mean_luminance_delta": row["mean_luminance"] - ref_metrics["mean_luminance"],
                }
            )
        if deltas:
            payload["reference_dir"] = args.reference_dir
            payload["delta_summary"] = summarize(deltas)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8-sig") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print(json.dumps({"output_json": args.output_json, "count": len(records)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
