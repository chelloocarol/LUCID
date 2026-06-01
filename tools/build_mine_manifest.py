#!/usr/bin/env python3
"""
Build a manifest for the current mine dehazing dataset.

Expected per-video layout:

per_video/
  video_name/
    filtered/
      input/
      target/
      mask/

The script creates:
- mine_manifest.csv
- mine_split.json
- mine_summary.json

Default split behavior:
- if there are 6 or more videos: 4 train, 1 val, 1 test
- otherwise: last video for test, second last for val when possible
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Sample:
    split: str
    video: str
    stem: str
    input_path: str
    target_path: str
    mask_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifests for the mine dehazing dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            r"D:\健全版\工具代码\视频抽帧-构建数据集-生成lut表一条龙\datasets\datasetB_quality_per_video\per_video"
        ),
        help="Root directory that contains one folder per video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\ARIS\COA\mine_research\data"),
        help="Directory to write manifest and split files into.",
    )
    parser.add_argument(
        "--train-videos",
        nargs="*",
        default=None,
        help="Explicit train video names. Overrides auto split when provided together with val/test lists.",
    )
    parser.add_argument(
        "--val-videos",
        nargs="*",
        default=None,
        help="Explicit validation video names.",
    )
    parser.add_argument(
        "--test-videos",
        nargs="*",
        default=None,
        help="Explicit test video names.",
    )
    return parser


def sorted_video_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def auto_split(video_names: List[str]) -> Dict[str, str]:
    if not video_names:
        raise ValueError("No video folders were found.")

    mapping: Dict[str, str] = {}
    if len(video_names) >= 6:
        train_names = video_names[:4]
        val_names = [video_names[4]]
        test_names = [video_names[5]]
        extra_names = video_names[6:]
        train_names.extend(extra_names)
    elif len(video_names) == 5:
        train_names = video_names[:3]
        val_names = [video_names[3]]
        test_names = [video_names[4]]
    elif len(video_names) == 4:
        train_names = video_names[:2]
        val_names = [video_names[2]]
        test_names = [video_names[3]]
    elif len(video_names) == 3:
        train_names = [video_names[0]]
        val_names = [video_names[1]]
        test_names = [video_names[2]]
    elif len(video_names) == 2:
        train_names = [video_names[0]]
        val_names = []
        test_names = [video_names[1]]
    else:
        train_names = [video_names[0]]
        val_names = []
        test_names = []

    for name in train_names:
        mapping[name] = "train"
    for name in val_names:
        mapping[name] = "val"
    for name in test_names:
        mapping[name] = "test"
    return mapping


def explicit_split(
    video_names: List[str],
    train_videos: Iterable[str] | None,
    val_videos: Iterable[str] | None,
    test_videos: Iterable[str] | None,
) -> Dict[str, str]:
    if not (train_videos or val_videos or test_videos):
        return auto_split(video_names)

    mapping: Dict[str, str] = {}
    for split_name, names in (
        ("train", train_videos or []),
        ("val", val_videos or []),
        ("test", test_videos or []),
    ):
        for name in names:
            if name not in video_names:
                raise ValueError(f"Unknown video name in split config: {name}")
            mapping[name] = split_name

    for name in video_names:
        mapping.setdefault(name, "train")
    return mapping


def pair_samples(video_dir: Path, split_name: str) -> Tuple[List[Sample], Dict[str, int]]:
    filtered_dir = video_dir / "filtered"
    input_dir = filtered_dir / "input"
    target_dir = filtered_dir / "target"
    mask_dir = filtered_dir / "mask"

    if not input_dir.exists() or not target_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Missing expected filtered directories under {video_dir}. "
            "Expected input/, target/, and mask/."
        )

    target_by_stem = {path.stem: path for path in target_dir.iterdir() if path.is_file()}
    mask_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}

    rows: List[Sample] = []
    missing_target = 0
    missing_mask = 0

    for input_path in sorted([path for path in input_dir.iterdir() if path.is_file()], key=lambda p: p.name):
        stem = input_path.stem
        target_path = target_by_stem.get(stem)
        mask_path = mask_by_stem.get(stem)
        if target_path is None:
            missing_target += 1
            continue
        if mask_path is None:
            missing_mask += 1
            continue

        rows.append(
            Sample(
                split=split_name,
                video=video_dir.name,
                stem=stem,
                input_path=str(input_path),
                target_path=str(target_path),
                mask_path=str(mask_path),
            )
        )

    stats = {
        "usable_samples": len(rows),
        "missing_target": missing_target,
        "missing_mask": missing_mask,
    }
    return rows, stats


def write_manifest(output_dir: Path, samples: List[Sample]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mine_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "video", "stem", "input_path", "target_path", "mask_path"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)
    return manifest_path


def write_json(output_path: Path, payload: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_dirs = sorted_video_dirs(args.input_root)
    video_names = [path.name for path in video_dirs]
    split_map = explicit_split(video_names, args.train_videos, args.val_videos, args.test_videos)

    samples: List[Sample] = []
    summary = {
        "input_root": str(args.input_root),
        "video_count": len(video_dirs),
        "videos": {},
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "sample_count": 0,
    }

    for video_dir in video_dirs:
        split_name = split_map[video_dir.name]
        rows, stats = pair_samples(video_dir, split_name)
        samples.extend(rows)
        summary["videos"][video_dir.name] = {
            "split": split_name,
            **stats,
        }
        summary["split_counts"][split_name] += len(rows)

    summary["sample_count"] = len(samples)

    manifest_path = write_manifest(args.output_dir, samples)
    split_path = args.output_dir / "mine_split.json"
    summary_path = args.output_dir / "mine_summary.json"

    write_json(
        split_path,
        {
            "video_to_split": split_map,
            "train_videos": [name for name, split in split_map.items() if split == "train"],
            "val_videos": [name for name, split in split_map.items() if split == "val"],
            "test_videos": [name for name, split in split_map.items() if split == "test"],
        },
    )
    write_json(summary_path, summary)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote split file: {split_path}")
    print(f"Wrote summary file: {summary_path}")
    print(json.dumps(summary["split_counts"], ensure_ascii=False))
    print(f"Total usable samples: {summary['sample_count']}")


if __name__ == "__main__":
    main()
