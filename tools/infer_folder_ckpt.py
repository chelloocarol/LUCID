import argparse
import glob
import json
import os
import sys

import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import LUCIDMine, Student, Student_x, Teacher


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


def list_images(input_dir):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_arch", default="student", choices=["student", "student_x", "teacher", "lucidmine"])
    parser.add_argument("--state_key", default="model", choices=["model", "ema_model"])
    parser.add_argument(
        "--output_ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image extension. Default: png (recommended for reproducible figure assets).",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_arch).to(device)
    state_dict = load_state_dict_from_checkpoint(args.checkpoint_path, args.state_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    images = list_images(args.input_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for image_path in images:
            haze = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            orig_h, orig_w = haze.shape[2], haze.shape[3]
            haze = Resize(
                (max(orig_h // 16 * 16, 16), max(orig_w // 16 * 16, 16)),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )(haze)
            output = model(haze)[0].clamp(0, 1).squeeze(0)
            output = Resize((orig_h, orig_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(output)
            stem, _ = os.path.splitext(os.path.basename(image_path))
            out_path = os.path.join(args.output_dir, f"{stem}.{args.output_ext}")
            torchvision.utils.save_image(output, out_path)

    print(
        json.dumps(
            {
                "checkpoint_path": args.checkpoint_path,
                "input_dir": args.input_dir,
                "output_dir": args.output_dir,
                "state_key": args.state_key,
                "image_count": len(images),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
