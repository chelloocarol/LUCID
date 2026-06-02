import argparse
import csv
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
from model import Student, Student_x, Teacher, LUCIDMine


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
    }


def reduce_metric_store(store):
    return {key: float(np.mean(values)) if values else 0.0 for key, values in store.items()}


def build_model(model_arch):
    if model_arch == "student":
        return Student()
    if model_arch == "student_x":
        return Student_x()
    if model_arch == "teacher":
        return Teacher()
    if model_arch == "lucidmine":
        return LUCIDMine()
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


def evaluate(model, loader, device):
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
            metric_row = {
                "masked_psnr": masked_psnr(pred, target, reliability).item(),
                "masked_ssim": (1.0 - masked_ssim_loss(pred, target, reliability)).item(),
                "masked_l1": masked_l1_loss(pred, target, reliability).item(),
                "full_psnr": masked_psnr(pred, target, None).item(),
                "full_ssim": ssim(pred, target).item(),
                "reliability": reliability.mean().item(),
            }

            video = batch["video"][0]
            split = batch["split"][0]
            key = f"{split}/{video}"
            for name, value in metric_row.items():
                overall[name].append(value)
                by_video[video][name].append(value)
                by_split[split][name].append(value)
                by_split_video[key][name].append(value)

    return overall, by_video, by_split, by_split_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--model_arch", default="student", choices=["student", "student_x", "teacher", "lucidmine"])
    parser.add_argument("--state_key", default="model", choices=["model", "ema_model"])
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_arch).to(device)
    state_dict = load_state_dict_from_checkpoint(args.checkpoint_path, args.state_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    dataset = MineManifestDataset(
        args.manifest_path,
        split=None,
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

    overall, by_video, by_split, by_split_video = evaluate(model, loader, device)
    video_metadata = load_video_metadata(args.manifest_path)
    payload = {
        "overall": reduce_metric_store(overall),
        "by_video": {key: reduce_metric_store(value) for key, value in sorted(by_video.items())},
        "by_split": {key: reduce_metric_store(value) for key, value in sorted(by_split.items())},
        "by_split_video": {key: reduce_metric_store(value) for key, value in sorted(by_split_video.items())},
        "aggregate_scopes": build_aggregate_scopes(by_split_video, video_metadata),
        "video_metadata": video_metadata,
        "checkpoint_path": args.checkpoint_path,
        "state_key": args.state_key,
        "device": device,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8-sig") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.output_csv:
        rows = []
        for key, metrics in payload["by_split_video"].items():
            split, video = key.split("/", 1)
            rows.append({"split": split, "video": video, **video_metadata[video], **metrics})
        with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"output_json": args.output_json, "groups": len(payload["by_split_video"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
