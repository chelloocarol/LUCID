import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import Student, Student_x, Teacher


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


def read_manifest_rows(manifest_path, split):
    manifest_path = Path(manifest_path)
    rows = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is None or row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {manifest_path} for split={split!r}")
    return rows


def run_model_on_pil(model, image, device, transform):
    with torch.no_grad():
        haze = transform(image).unsqueeze(0).to(device)
        orig_h, orig_w = haze.shape[2], haze.shape[3]
        resized_h = max(orig_h // 16 * 16, 16)
        resized_w = max(orig_w // 16 * 16, 16)
        haze = Resize((resized_h, resized_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)
        pred = model(haze)[0].clamp(0, 1).squeeze(0)
        pred = Resize((orig_h, orig_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(pred)
    return pred.cpu()


def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.clamp(0, 1)
    return to_pil_image(image_tensor)


def make_triptych(input_pil, option_a_pil, option_b_pil):
    tile_w = max(input_pil.width, option_a_pil.width, option_b_pil.width)
    tile_h = max(input_pil.height, option_a_pil.height, option_b_pil.height)
    label_h = 28
    gutter = 12
    canvas = Image.new("RGB", (tile_w * 3 + gutter * 4, tile_h + label_h + gutter * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    slots = [("Input", input_pil), ("Option A", option_a_pil), ("Option B", option_b_pil)]
    for idx, (label, tile) in enumerate(slots):
        x0 = gutter + idx * (tile_w + gutter)
        y0 = gutter + label_h
        canvas.paste(tile, (x0 + (tile_w - tile.width) // 2, y0 + (tile_h - tile.height) // 2))
        draw.text((x0, 6), label, fill=(0, 0, 0))

    return canvas


def save_png(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(0, 1), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--baseline_state_key", default="model", choices=["model", "ema_model"])
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--adapted_ckpt", required=True)
    parser.add_argument("--adapted_state_key", default="ema_model", choices=["model", "ema_model"])
    parser.add_argument("--adapted_label", default="adapted")
    parser.add_argument("--model_arch", default="student", choices=["student", "student_x", "teacher"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest_rows(args.manifest_path, args.split)
    rng = random.Random(args.seed)
    chosen_rows = rows if len(rows) <= args.sample_count else rng.sample(rows, args.sample_count)
    chosen_rows = sorted(chosen_rows, key=lambda row: (row["video"], row["stem"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    baseline = build_model(args.model_arch).to(device)
    baseline_state = load_state_dict_from_checkpoint(args.baseline_ckpt, args.baseline_state_key)
    baseline.load_state_dict(baseline_state, strict=False)
    baseline.eval()

    adapted = build_model(args.model_arch).to(device)
    adapted_state = load_state_dict_from_checkpoint(args.adapted_ckpt, args.adapted_state_key)
    adapted.load_state_dict(adapted_state, strict=False)
    adapted.eval()

    answer_rows = []
    for sample_index, row in enumerate(chosen_rows, start=1):
        input_path = Path(row["input_path"])
        image = Image.open(input_path).convert("RGB")

        baseline_pred = run_model_on_pil(baseline, image, device, transform)
        adapted_pred = run_model_on_pil(adapted, image, device, transform)

        if rng.random() < 0.5:
            option_a_tensor = baseline_pred
            option_b_tensor = adapted_pred
            option_a_model = args.baseline_label
            option_b_model = args.adapted_label
        else:
            option_a_tensor = adapted_pred
            option_b_tensor = baseline_pred
            option_a_model = args.adapted_label
            option_b_model = args.baseline_label

        sample_name = f"{sample_index:03d}_{row['video']}_{row['stem']}"
        sample_dir = samples_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        image.save(sample_dir / "input.png")
        save_png(option_a_tensor, sample_dir / "option_A.png")
        save_png(option_b_tensor, sample_dir / "option_B.png")

        panel = make_triptych(image, tensor_to_pil(option_a_tensor), tensor_to_pil(option_b_tensor))
        panel.save(sample_dir / "panel.png")

        answer_rows.append(
            {
                "sample_name": sample_name,
                "split": row["split"],
                "video": row["video"],
                "stem": row["stem"],
                "input_path": str(input_path),
                "option_A_model": option_a_model,
                "option_B_model": option_b_model,
            }
        )

    answer_key_path = output_dir / "answer_key.csv"
    with answer_key_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_name",
                "split",
                "video",
                "stem",
                "input_path",
                "option_A_model",
                "option_B_model",
            ],
        )
        writer.writeheader()
        writer.writerows(answer_rows)

    protocol = {
        "manifest_path": args.manifest_path,
        "split": args.split,
        "sample_count": len(answer_rows),
        "seed": args.seed,
        "baseline_ckpt": args.baseline_ckpt,
        "baseline_state_key": args.baseline_state_key,
        "adapted_ckpt": args.adapted_ckpt,
        "adapted_state_key": args.adapted_state_key,
        "blinding_rule": "Option A and Option B are randomized independently for each sample. Use answer_key.csv only after ratings are completed.",
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": len(answer_rows),
                "answer_key": str(answer_key_path),
                "protocol": str(output_dir / "protocol.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
