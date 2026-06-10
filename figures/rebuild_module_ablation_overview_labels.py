from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "qualitative_module_ablation"

SCENE_IDS = [
    "71016ff12f9dd8c27bcd77a6f10a65d5",
    "67c0c50c50e7f2ebfb516aece2685685",
    "802bb080690ffaa6433be446cdb0fc4f",
    "b8cb82765f04a1696f41134bf27a144d",
]

COLUMNS = [
    ("Backbone", "input.jpg"),
    ("+VCA", "vca.jpg"),
    ("+GARC", "garc.jpg"),
    ("LUCIDMine", "lucid_neural.jpg"),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(Image.open(path).convert("RGB"), size, method=Image.Resampling.LANCZOS)


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=fnt, fill=(20, 20, 20))


def main() -> None:
    cell_w, cell_h = 205, 118
    header_h = 34
    gap = 3
    width = len(COLUMNS) * cell_w + (len(COLUMNS) - 1) * gap
    height = header_h + len(SCENE_IDS) * cell_h + (len(SCENE_IDS) - 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = font(18, bold=True)

    for col, (label, _) in enumerate(COLUMNS):
        x = col * (cell_w + gap)
        centered_text(draw, (x, 0, x + cell_w, header_h), label, label_font)

    for row, scene_id in enumerate(SCENE_IDS):
        y = header_h + row * (cell_h + gap)
        for col, (_, filename) in enumerate(COLUMNS):
            x = col * (cell_w + gap)
            source = FIG_DIR / scene_id / filename
            if not source.exists():
                raise FileNotFoundError(source)
            canvas.paste(fit_tile(source, (cell_w, cell_h)), (x, y))

    out = FIG_DIR / "user_four_module_ablation_overview_lucid_neural_fixed.png"
    canvas.save(out, dpi=(300, 300), quality=98)
    print(out)


if __name__ == "__main__":
    main()
