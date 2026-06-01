import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import MineManifestDataset
from loss import BlendIgnoreSSIMLoss, MaskedL1Loss, SSIM, masked_psnr
from metric import ssim
from model import Student, Student_x, Teacher


LOW_COUNT_THRESHOLD = 15


def bucket_for_count(pair_count):
    if pair_count <= 15:
        return "1-15"
    if pair_count <= 30:
        return "16-30"
    if pair_count <= 45:
        return "31-45"
    return "46-64"


def init_metric_store():
    return {
        "masked_psnr": [],
        "masked_ssim": [],
        "masked_l1": [],
        "full_psnr": [],
        "full_ssim": [],
        "reliability": [],
        "mean_luminance": [],
        "clipped_ratio": [],
        "chroma_blockiness": [],
        "lowfreq_chroma_error": [],
    }


def reduce_metric_store(store):
    return {key: float(np.mean(values)) if values else 0.0 for key, values in store.items()}


def _normalize_kernel_size(kernel_size):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _reflect_avg_pool(tensor, kernel_size):
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return tensor
    pad = kernel_size // 2
    tensor = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    return torch.nn.functional.avg_pool2d(tensor, kernel_size=kernel_size, stride=1)


def luminance(image):
    return 0.2126 * image[:, 0:1, :, :] + 0.7152 * image[:, 1:2, :, :] + 0.0722 * image[:, 2:3, :, :]


def clipped_ratio(image, low=0.01, high=0.99):
    clipped = ((image <= low) | (image >= high)).any(dim=1, keepdim=True).float()
    return clipped.flatten(1).mean(dim=1)


def rgb_to_ycbcr(image):
    r = image[:, 0:1, :, :]
    g = image[:, 1:2, :, :]
    b = image[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def _crop_to_blocks(image, block_size):
    height = image.shape[2] // block_size * block_size
    width = image.shape[3] // block_size * block_size
    if height == 0 or width == 0:
        return image
    return image[:, :, :height, :width]


def chroma_blockiness(image, block_size=32):
    ycbcr = rgb_to_ycbcr(image.clamp(0, 1))[:, 1:, :, :]
    ycbcr = _crop_to_blocks(ycbcr, block_size)
    if ycbcr.shape[2] < block_size or ycbcr.shape[3] < block_size:
        return torch.zeros(ycbcr.shape[0], device=image.device, dtype=image.dtype)
    pooled = torch.nn.functional.avg_pool2d(ycbcr, kernel_size=block_size, stride=block_size)
    diffs = []
    if pooled.shape[3] > 1:
        diffs.append((pooled[:, :, :, 1:] - pooled[:, :, :, :-1]).abs())
    if pooled.shape[2] > 1:
        diffs.append((pooled[:, :, 1:, :] - pooled[:, :, :-1, :]).abs())
    if not diffs:
        return torch.zeros(pooled.shape[0], device=image.device, dtype=image.dtype)
    flat = torch.cat([diff.flatten(1) for diff in diffs], dim=1)
    return flat.mean(dim=1)


def lowfreq_chroma_error(pred, target, kernel_size=11):
    pred_low = _reflect_avg_pool(rgb_to_ycbcr(pred.clamp(0, 1))[:, 1:, :, :], kernel_size)
    target_low = _reflect_avg_pool(rgb_to_ycbcr(target.clamp(0, 1))[:, 1:, :, :], kernel_size)
    return (pred_low - target_low).abs().flatten(1).mean(dim=1)


def build_model(model_arch):
    if model_arch == "student":
        return Student()
    if model_arch == "student_x":
        return Student_x()
    if model_arch == "teacher":
        return Teacher()
    raise ValueError(f"Unsupported model arch: {model_arch}")


def load_state_dict_from_checkpoint(checkpoint_path, state_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and state_key in checkpoint and isinstance(checkpoint[state_key], dict):
        checkpoint = checkpoint[state_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

    clean_state = {}
    for key, value in checkpoint.items():
        clean_state[key.replace("module.", "")] = value
    return clean_state


def load_video_metadata(manifest_path):
    dataset = MineManifestDataset(manifest_path, split=None, train=False, crop_size=256)
    grouped = defaultdict(list)
    for row in dataset.records:
        grouped[row["video"]].append(row)

    metadata = {}
    for video, rows in grouped.items():
        pair_count = len(rows)
        metadata[video] = {
            "split": rows[0]["split"],
            "pair_count": pair_count,
            "bucket": bucket_for_count(pair_count),
            "is_low_count": pair_count <= LOW_COUNT_THRESHOLD,
        }
    return metadata


def build_aggregate_scopes(by_split_video, video_metadata):
    scopes = {}
    for scope_name, include_low_count in (("full_eval", True), ("primary_eval", False)):
        overall = init_metric_store()
        by_split = defaultdict(init_metric_store)
        selected_videos = set()
        for key, store in by_split_video.items():
            split, video = key.split("/", 1)
            if not include_low_count and video_metadata[video]["is_low_count"]:
                continue
            selected_videos.add(video)
            for metric_name, values in store.items():
                overall[metric_name].extend(values)
                by_split[split][metric_name].extend(values)
        scopes[scope_name] = {
            "overall": reduce_metric_store(overall),
            "by_split": {split: reduce_metric_store(store) for split, store in sorted(by_split.items())},
            "video_count": len(selected_videos),
            "videos": sorted(selected_videos),
        }
    return scopes


def evaluate(model, loader, device, artifact_block_size, artifact_pool_kernel):
    masked_l1_loss = MaskedL1Loss().to(device)
    masked_ssim_loss = BlendIgnoreSSIMLoss(SSIM().to(device)).to(device)
    overall = init_metric_store()
    by_video = defaultdict(init_metric_store)
    by_split = defaultdict(init_metric_store)
    by_split_video = defaultdict(init_metric_store)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            haze = batch["haze"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            reliability = batch["reliability"].to(device, non_blocking=True)

            pred = model(haze)[0].clamp(0, 1)
            metrics = {
                "masked_psnr": masked_psnr(pred, target, reliability).item(),
                "masked_ssim": (1.0 - masked_ssim_loss(pred, target, reliability)).item(),
                "masked_l1": masked_l1_loss(pred, target, reliability).item(),
                "full_psnr": masked_psnr(pred, target, None).item(),
                "full_ssim": ssim(pred, target).item(),
                "reliability": reliability.mean().item(),
                "mean_luminance": luminance(pred).mean().item(),
                "clipped_ratio": clipped_ratio(pred).mean().item(),
                "chroma_blockiness": chroma_blockiness(pred, block_size=artifact_block_size).mean().item(),
                "lowfreq_chroma_error": lowfreq_chroma_error(pred, target, kernel_size=artifact_pool_kernel).mean().item(),
            }

            split = batch["split"][0]
            video = batch["video"][0]
            split_video = f"{split}/{video}"
            for name, value in metrics.items():
                overall[name].append(value)
                by_video[video][name].append(value)
                by_split[split][name].append(value)
                by_split_video[split_video][name].append(value)

    return overall, by_video, by_split, by_split_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_arch", default="student", choices=["student", "student_x", "teacher"])
    parser.add_argument("--state_key", default="model", choices=["model", "ema_model"])
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--artifact_block_size", type=int, default=32)
    parser.add_argument("--artifact_pool_kernel", type=int, default=11)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_arch).to(device)
    state_dict = load_state_dict_from_checkpoint(args.checkpoint_path, args.state_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    dataset = MineManifestDataset(
        args.manifest_path,
        split=args.split if args.split else None,
        train=False,
        crop_size=args.crop_size,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    overall, by_video, by_split, by_split_video = evaluate(
        model,
        loader,
        device,
        artifact_block_size=args.artifact_block_size,
        artifact_pool_kernel=args.artifact_pool_kernel,
    )
    video_metadata = load_video_metadata(args.manifest_path)
    payload = {
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "state_key": args.state_key,
        "device": device,
        "artifact_policy": {
            "block_size": args.artifact_block_size,
            "lowfreq_pool_kernel": _normalize_kernel_size(args.artifact_pool_kernel),
            "clipped_ratio_range": [0.01, 0.99],
            "mean_luminance": "predicted RGB luminance mean",
            "chroma_blockiness": "adjacent Cb/Cr block-mean absolute difference; lower is better",
            "lowfreq_chroma_error": "low-pass Cb/Cr L1 error against target; lower is better",
        },
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "metrics": reduce_metric_store(overall),
        "overall": reduce_metric_store(overall),
        "by_video": {key: reduce_metric_store(store) for key, store in sorted(by_video.items())},
        "by_split": {key: reduce_metric_store(store) for key, store in sorted(by_split.items())},
        "by_split_video": {key: reduce_metric_store(store) for key, store in sorted(by_split_video.items())},
        "video_metadata": video_metadata,
        "aggregate_scopes": build_aggregate_scopes(by_split_video, video_metadata),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8-sig") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
