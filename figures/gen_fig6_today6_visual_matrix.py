from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SOURCE_DIR = ROOT / "data"
WORK_DIR = ROOT / "figure_data" / "images" / "fig6_today6"
INPUT_DIR = WORK_DIR / "input"

SCENES = [
    "e9c7e537ea96d384fdafe31a744781d7.jpg",
    "c72d1ab6361e36ae9a1b15120b054813.jpg",
    "b9ae1c4e4bab635134d527b1cab8b0cc.jpg",
    "8d1a02be508cb561ac873fe18c4b4861.jpg",
    "a47ac21e3d7ea521c6f963e52137b6a8.jpg",
    "227.jpg",
]

METHODS = [
    ("Input", "input"),
    ("DCP", "dcp"),
    ("CLAHE", "clahe"),
    ("Retinex", "retinex"),
    ("AdaIR", "adair"),
    ("LUCIDMine", "lucidmine"),
]


def font(size: int = 22, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:/Windows/Fonts/timesbd.ttf" if bold else r"C:/Windows/Fonts/times.ttf",
        r"C:/Windows/Fonts/arialbd.ttf" if bold else r"C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def np_img(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255.0 + 0.5), "RGB")


def dcp_like(image: Image.Image) -> Image.Image:
    arr = np_img(image)
    dark = arr.min(axis=2)
    airlight = np.percentile(arr.reshape(-1, 3), 99, axis=0)
    transmission = 1.0 - 0.72 * dark
    transmission = np.clip(transmission[..., None], 0.28, 1.0)
    restored = np.clip((arr - airlight * (1 - transmission)) / transmission, 0, 1)
    return ImageEnhance.Contrast(to_image(restored)).enhance(1.12)


def clahe_like(image: Image.Image) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.autocontrast(y, cutoff=1)
    y = ImageOps.equalize(y)
    merged = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
    return ImageEnhance.Color(ImageEnhance.Contrast(merged).enhance(1.08)).enhance(0.95)


def retinex_like(image: Image.Image) -> Image.Image:
    arr = np_img(image)
    eps = 1e-3
    logs = []
    for radius in [9, 31, 91]:
        blur = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")).astype(np.float32) / 255.0
        logs.append(np.log(arr + eps) - np.log(blur + eps))
    ret = np.mean(logs, axis=0)
    lo, hi = np.percentile(ret, [1, 99])
    ret = (ret - lo) / max(float(hi - lo), eps)
    return ImageEnhance.Color(to_image(ret)).enhance(0.82)


def save_all(image: Image.Image, stem: str) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    svg = FIGURES / f"{stem}.svg"
    image.save(png)
    image.convert("RGB").save(pdf, "PDF", resolution=300)
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">\n'
        f'  <image width="{image.width}" height="{image.height}" href="data:image/png;base64,{encoded}"/>\n'
        f"</svg>\n",
        encoding="utf-8",
    )


def prepare_inputs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        src = SOURCE_DIR / scene
        if not src.exists():
            raise FileNotFoundError(src)
        dst = INPUT_DIR / scene
        if not dst.exists():
            Image.open(src).convert("RGB").save(dst)


def generate_classical_outputs() -> None:
    output_map = {
        "dcp": dcp_like,
        "clahe": clahe_like,
        "retinex": retinex_like,
    }
    for method, fn in output_map.items():
        out_dir = WORK_DIR / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for scene in SCENES:
            out_path = out_dir / scene
            if out_path.exists():
                continue
            image = Image.open(INPUT_DIR / scene).convert("RGB")
            fn(image).save(out_path)


def generate_adair_outputs() -> None:
    out_dir = WORK_DIR / "adair"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = ROOT / "scripts" / "run_adair_single_image.py"
    missing = [scene for scene in SCENES if not (out_dir / scene).exists()]
    for scene in missing:
        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(INPUT_DIR / scene),
            "--output",
            str(out_dir / scene),
            "--cpu",
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))


def verify_lucid_outputs() -> None:
    out_dir = WORK_DIR / "lucidmine"
    missing = [scene for scene in SCENES if not (out_dir / scene).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing LUCIDMine outputs. Run tools/infer_folder_ckpt.py with "
            f"experiment/LUCIDMine/modal_run_v2/best.pth first: {missing}"
        )


def make_matrix() -> Image.Image:
    cell_w, cell_h = 228, 136
    gap = 2
    row_gap = 2
    margin = 6
    header_h = 36
    width = margin * 2 + cell_w * len(METHODS) + gap * (len(METHODS) - 1)
    height = margin * 2 + header_h + cell_h * len(SCENES) + row_gap * (len(SCENES) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(22, True)
    for col, (label, _) in enumerate(METHODS):
        x = margin + col * (cell_w + gap)
        bbox = draw.textbbox((0, 0), label, font=header_font)
        draw.text(
            (x + (cell_w - (bbox[2] - bbox[0])) / 2, margin),
            label,
            font=header_font,
            fill=(24, 30, 38),
        )

    for row, scene in enumerate(SCENES):
        for col, (_, key) in enumerate(METHODS):
            path = WORK_DIR / key / scene
            tile = ImageOps.fit(Image.open(path).convert("RGB"), (cell_w, cell_h), method=Image.Resampling.LANCZOS)
            x = margin + col * (cell_w + gap)
            y = margin + header_h + row * (cell_h + row_gap)
            canvas.paste(tile, (x, y))
    return canvas


def main() -> None:
    prepare_inputs()
    generate_classical_outputs()
    generate_adair_outputs()
    verify_lucid_outputs()
    matrix = make_matrix()
    save_all(matrix, "fig6_today6_6x6_visual_matrix")
    manifest = {
        "figure": "fig6_today6_6x6_visual_matrix",
        "rows": SCENES,
        "columns": [label for label, _ in METHODS],
        "source": "main branch data images added in commit a59a979/6d5d54c",
        "lucidmine_checkpoint": "experiment/LUCIDMine/modal_run_v2/best.pth from origin/claude/lucidmine-paper-audit-fzyBW",
        "layout": "Compact white-gutter matrix with 2 px inter-image gaps, matching the narrow-spacing style used by Fig.5.",
    }
    (ROOT / "figure_data" / "fig6_today6_6x6_visual_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
