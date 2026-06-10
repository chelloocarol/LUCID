from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_adair_repo(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    from net.model import AdaIR  # type: ignore

    return AdaIR


def load_model(repo_dir: Path, ckpt_path: Path, device: torch.device):
    AdaIR = load_adair_repo(repo_dir)
    model = AdaIR(decoder=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {
        key.removeprefix("net."): value
        for key, value in state_dict.items()
        if key.startswith("net.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys: {len(unexpected)}")
    if missing:
        print(f"Missing checkpoint keys: {len(missing)}")
    model.to(device).eval()
    return model


def pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)
    if pad_h or pad_w:
        x = F.pad(x, pad, mode="reflect")
    return x, height, width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(r"D:\ARIS\external\AdaIR"))
    parser.add_argument("--ckpt", type=Path, default=Path(r"D:\ARIS\external\AdaIR\ckpt\adair-single-dehaze.ckpt"))
    parser.add_argument("--input", type=Path, default=Path(r"D:\ARIS\_github_upload_LUCID_20260601\data\223.jpeg"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\ARIS\_github_upload_LUCID_20260601\figure_data\images\cross_domain_223\adair_official_dehaze.png"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(args.repo, args.ckpt, device)

    image = Image.open(args.input).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    tensor, height, width = pad_to_multiple(tensor, multiple=8)

    with torch.no_grad():
        restored = model(tensor).clamp(0, 1)
    restored = restored[:, :, :height, :width]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_image = transforms.ToPILImage()(restored.squeeze(0).cpu())
    out_image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
