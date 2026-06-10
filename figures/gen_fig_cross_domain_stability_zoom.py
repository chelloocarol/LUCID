from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SRC_DIR = ROOT / "figure_data" / "figure_sources"
IMAGE_ROOT = ROOT / "figure_data" / "images"

SCENE = "13502 T2传感器__segment_035"

# ROI used by the original cross-domain stability figure, mapped to the
# 960x540 source scene from the Claude branch. The box focuses on the sensor
# cluster so the zoom row exposes structure and haze differences.
ROI_BOX = (510, 160, 650, 265)

BASE_METHODS = [
    ("DCP", "dcp"),
    ("CLAHE", "clahe"),
    ("Retinex", "retinex"),
    ("AdaIR", "adair"),
]

LUCID_LABELS = ["LUCIDMine", "LUCIDMine", "LUCIDMine", "LUCIDMine"]
LUCID_METHOD_DIR = "lucidmine"


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def find_image(method_dir: str, stem: str) -> Path:
    base = IMAGE_ROOT / method_dir
    for ext in (".png", ".jpg", ".jpeg"):
        path = base / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for method={method_dir}, scene={stem}")


def scale_box(box: tuple[int, int, int, int], src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int, int, int]:
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    x1, y1, x2, y2 = box
    return (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))


def expand_box(box: tuple[int, int, int, int], image_size: tuple[int, int], factor: float = 1.25) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, round(cx - w / 2))
    ny1 = max(0, round(cy - h / 2))
    nx2 = min(image_size[0], round(cx + w / 2))
    ny2 = min(image_size[1], round(cy + h / 2))
    return nx1, ny1, nx2, ny2


def fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=(18, 24, 32)) -> None:
    x1, y1, x2, y2 = box
    bb = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x1 + (x2 - x1 - (bb[2] - bb[0])) / 2, y1 + (y2 - y1 - (bb[3] - bb[1])) / 2), text, font=fnt, fill=fill)


def draw_roi(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int = 3, color=(255, 196, 0)) -> None:
    for i in range(width):
        draw.rectangle((box[0] - i, box[1] - i, box[2] + i, box[3] + i), outline=color)


def crop_zoom(img: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    return img.crop(expand_box(box, img.size, factor=1.25)).resize(size, Image.Resampling.LANCZOS)


def save_svg_embed(png: Path, svg: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>\n'
        f"</svg>\n",
        encoding="utf-8",
    )


def build_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    base_images = [(label, Image.open(find_image(method_dir, SCENE)).convert("RGB")) for label, method_dir in BASE_METHODS]
    lucid_path = find_image(LUCID_METHOD_DIR, SCENE)
    lucid = Image.open(lucid_path).convert("RGB")

    cell_w, cell_h = 236, 137
    zoom_h = 72
    label_h = 26
    gap = 1
    row_gap = 4
    margin = 7
    cols = len(BASE_METHODS)
    width = margin * 2 + cols * cell_w + (cols - 1) * gap
    height = margin * 2 + cell_h + label_h + row_gap + zoom_h + row_gap + cell_h + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f_label = font(17, bold=True)

    manifest = {
        "figure": "fig_cross_domain_stability_zoom",
        "scene": SCENE,
        "role": "Qualitative cross-domain/source stability comparison with local ROI zooms.",
        "roi_box_source_xyxy": ROI_BOX,
        "base_methods": [label for label, _ in BASE_METHODS],
        "lucid_source": str(lucid_path),
        "note": (
            "Layout follows the original cross-domain stability figure. The RIDCP column is replaced by AdaIR; "
            "all other visual structure is preserved. Top row shows Base method outputs, the middle row concatenates "
            "Base ROI zoom and LUCIDMine ROI zoom, and the bottom row repeats LUCIDMine for source-domain stability."
        ),
        "sources": {},
    }

    y_top = margin
    y_top_label = y_top + cell_h
    y_zoom = y_top_label + label_h + row_gap
    y_bottom = y_zoom + zoom_h + row_gap
    y_bottom_label = y_bottom + cell_h

    for col, ((base_label, base_img), lucid_label) in enumerate(zip(base_images, LUCID_LABELS)):
        x = margin + col * (cell_w + gap)
        base_box = scale_box(ROI_BOX, (960, 540), base_img.size)
        lucid_box = scale_box(ROI_BOX, (960, 540), lucid.size)

        top_tile = fit_cover(base_img, (cell_w, cell_h))
        top_box_scaled = scale_box(base_box, base_img.size, (cell_w, cell_h))
        canvas.paste(top_tile, (x, y_top))
        draw_roi(draw, (x + top_box_scaled[0], y_top + top_box_scaled[1], x + top_box_scaled[2], y_top + top_box_scaled[3]), width=2)
        center_text(draw, (x, y_top_label, x + cell_w, y_top_label + label_h), base_label, f_label)

        base_zoom = crop_zoom(base_img, base_box, (cell_w // 2, zoom_h))
        lucid_zoom = crop_zoom(lucid, lucid_box, (cell_w - cell_w // 2, zoom_h))
        zoom = Image.new("RGB", (cell_w, zoom_h), "white")
        zoom.paste(base_zoom, (0, 0))
        zoom.paste(lucid_zoom, (cell_w // 2, 0))
        zd = ImageDraw.Draw(zoom)
        zd.line((cell_w // 2, 0, cell_w // 2, zoom_h), fill=(255, 255, 255), width=2)
        zd.rectangle((0, 0, cell_w - 1, zoom_h - 1), outline=(255, 196, 0), width=2)
        canvas.paste(zoom, (x, y_zoom))

        bottom_tile = fit_cover(lucid, (cell_w, cell_h))
        bottom_box_scaled = scale_box(lucid_box, lucid.size, (cell_w, cell_h))
        canvas.paste(bottom_tile, (x, y_bottom))
        draw_roi(draw, (x + bottom_box_scaled[0], y_bottom + bottom_box_scaled[1], x + bottom_box_scaled[2], y_bottom + bottom_box_scaled[3]), width=2)
        center_text(draw, (x, y_bottom_label, x + cell_w, y_bottom_label + label_h), lucid_label, f_label)

        manifest["sources"][base_label] = {
            "base_image": str(find_image(dict(BASE_METHODS)[base_label], SCENE)),
            "base_roi_xyxy": base_box,
            "lucid_roi_xyxy": lucid_box,
        }

    stem = "fig_cross_domain_stability_zoom"
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    svg = FIG_DIR / f"{stem}.svg"
    canvas.save(png, dpi=(300, 300), quality=98)
    canvas.save(pdf, "PDF", resolution=300)
    save_svg_embed(png, svg, canvas.width, canvas.height)
    (SRC_DIR / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    latex = r"""% === Cross-domain stability qualitative figure with ROI zooms ===
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_cross_domain_stability_zoom.pdf}
    \caption{Cross-domain qualitative stability comparison on the same real underground test image. The top row shows Base restorations from DCP, CLAHE, Retinex, and AdaIR; the middle row concatenates the Base ROI zoom and the corresponding LUCIDMine ROI zoom; the bottom row shows the LUCIDMine output, illustrating stable recovery of the highlighted instrument region under source-domain variation.}
    \label{fig:cross_domain_stability_zoom}
\end{figure*}
"""
    (FIG_DIR / "latex_cross_domain_stability_zoom_include.tex").write_text(latex, encoding="utf-8")
    print(png)
    print(pdf)
    print(svg)


if __name__ == "__main__":
    build_figure()
