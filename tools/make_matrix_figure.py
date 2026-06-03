"""
Publication-quality 5-scene × 6-method comparison matrix figure.
Columns: Input | DCP | CLAHE | Retinex | AdaIR | LUCIDMine (Ours)
Rows: 5 coal-mine scenes with English labels
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── layout ──────────────────────────────────────────────────────────────────
CELL_W      = 370       # px per image cell
CELL_H      = 208       # px per image cell (≈ 16:9)
GAP         = 3         # gap between cells (px)
HEADER_H    = 46        # column header row height
ROW_LABEL_W = 128       # left gutter for scene labels
BORDER      = 10        # outer white border
BG          = (255, 255, 255)

SCENES = [
    "Cable Side Glare",
    "Dust Scattering",
    "Miner Scene",
    "Instrument Haze",
    "Glare Saturation",
]
IDS    = ["221", "223", "224", "225", "226"]

METHODS = [
    ("Input",            False),
    ("DCP [1]",          False),
    ("CLAHE [4]",        False),
    ("Retinex [3]",      False),
    ("AdaIR [5]",        False),
    ("LUCIDMine\n(Ours)",True),
]

DIRS = {
    "Input":            "/tmp/user_input",
    "DCP [1]":          "/tmp/user_out/dcp",
    "CLAHE [4]":        "/tmp/user_out/clahe",
    "Retinex [3]":      "/tmp/user_out/retinex",
    "AdaIR [5]":        "/tmp/user_out/adair",
    "LUCIDMine\n(Ours)":"/tmp/user_out/lucidmine",
}

# colour palette
HEADER_BG_NORMAL = (240, 242, 244)
HEADER_BG_OURS   = (220, 237, 220)   # light green tint for LUCIDMine
HEADER_FG_NORMAL = (40,  40,  40)
HEADER_FG_OURS   = (20,  100, 20)
ROW_LABEL_BG     = (250, 250, 250)
ROW_LABEL_FG     = (30,  30,  30)
SEPARATOR_CLR    = (200, 200, 200)

CROP_TOP    = 22   # pixels to strip from the top (timestamp/overlay)
CROP_BOTTOM = 0    # pixels to strip from the bottom

# ── font helpers ─────────────────────────────────────────────────────────────
def _font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _font_reg(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# ── image loading ─────────────────────────────────────────────────────────────
def load_cell(method_label, img_id):
    d = DIRS[method_label]
    for ext in ("png", "jpeg", "jpg"):
        p = os.path.join(d, f"{img_id}.{ext}")
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            # crop overlays
            w, h = img.size
            top = min(CROP_TOP, h // 10)
            bot = h - CROP_BOTTOM if CROP_BOTTOM else h
            img = img.crop((0, top, w, bot))
            # resize to cell dimensions
            img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
            return img
    raise FileNotFoundError(f"No image for method={method_label!r} id={img_id!r} in {d}")

# ── draw centred text (multi-line) ─────────────────────────────────────────
def draw_text_centred(draw, text, box, font, fill):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    line_h = font.getbbox("Ag")[3] + 3
    total_h = len(lines) * line_h - 3
    y_start = y0 + (y1 - y0 - total_h) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = x0 + (x1 - x0 - tw) // 2
        draw.text((x, y_start + i * line_h), line, font=font, fill=fill)

def draw_text_centred_rotated(canvas, text, box, font, fill):
    """Draw vertically-centred, rotated 90° text inside box."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    # measure on a temp surface
    tmp = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    td  = ImageDraw.Draw(tmp)
    bbox = td.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # render horizontally then rotate
    txt_img = Image.new("RGBA", (tw + 4, th + 4), (255, 255, 255, 0))
    ti_draw = ImageDraw.Draw(txt_img)
    ti_draw.text((2, 2), text, font=font, fill=fill)
    txt_rot = txt_img.rotate(90, expand=True)
    rw, rh = txt_rot.size
    px = x0 + (bw - rw) // 2
    py = y0 + (bh - rh) // 2
    canvas.paste(txt_rot, (px, py), txt_rot)

# ── compose ──────────────────────────────────────────────────────────────────
def build():
    n_rows   = len(SCENES)
    n_cols   = len(METHODS)
    grid_w   = ROW_LABEL_W + n_cols * CELL_W + (n_cols - 1) * GAP
    grid_h   = HEADER_H    + n_rows * CELL_H + (n_rows - 1) * GAP
    total_w  = grid_w + 2 * BORDER
    total_h  = grid_h + 2 * BORDER

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    f_hdr  = _font(15)
    f_lbl  = _font_reg(13)

    # ── column headers ────────────────────────────────────────────────────
    for ci, (mname, is_ours) in enumerate(METHODS):
        x0 = BORDER + ROW_LABEL_W + ci * (CELL_W + GAP)
        y0 = BORDER
        x1 = x0 + CELL_W
        y1 = y0 + HEADER_H
        bg = HEADER_BG_OURS if is_ours else HEADER_BG_NORMAL
        fg = HEADER_FG_OURS if is_ours else HEADER_FG_NORMAL
        draw.rectangle([x0, y0, x1, y1], fill=bg)
        draw_text_centred(draw, mname, (x0, y0, x1, y1), f_hdr, fg)

    # optional: thin horizontal separator under header
    sep_y = BORDER + HEADER_H
    draw.line([(BORDER, sep_y), (total_w - BORDER, sep_y)], fill=SEPARATOR_CLR, width=1)

    # ── rows ──────────────────────────────────────────────────────────────
    for ri, (scene, img_id) in enumerate(zip(SCENES, IDS)):
        y0_cell = BORDER + HEADER_H + ri * (CELL_H + GAP)

        # row label gutter
        lx0 = BORDER
        lx1 = BORDER + ROW_LABEL_W - GAP
        ly1 = y0_cell + CELL_H
        draw.rectangle([lx0, y0_cell, lx1, ly1], fill=ROW_LABEL_BG)
        # thin right border
        draw.line([(lx1, y0_cell), (lx1, ly1)], fill=SEPARATOR_CLR, width=1)
        draw_text_centred_rotated(canvas, scene, (lx0, y0_cell, lx1, ly1), f_lbl, ROW_LABEL_FG)

        # image cells
        for ci, (mname, _) in enumerate(METHODS):
            cx0 = BORDER + ROW_LABEL_W + ci * (CELL_W + GAP)
            cell = load_cell(mname, img_id)
            canvas.paste(cell, (cx0, y0_cell))

    return canvas

if __name__ == "__main__":
    out_path = "/tmp/comparison_matrix.png"
    fig = build()
    fig.save(out_path, dpi=(300, 300))
    print(f"Saved → {out_path}  ({fig.size[0]}×{fig.size[1]} px)")
