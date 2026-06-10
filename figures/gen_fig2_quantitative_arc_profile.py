from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "figure_data" / "metrics" / "comparison_lowres_448x256.json"
FALLBACK_DATA_PATH = ROOT / "data" / "comparison_lowres_448x256.json"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "figure_data" / "tables"
OUT_STEM = "fig2_quantitative_arc_profile"

METHOD_ALIASES = {
    "Input": ("input", "Input"),
    "DCP": ("DCP",),
    "CLAHE": ("CLAHE",),
    "Retinex": ("Retinex",),
    "AdaIR": ("AdaIR", "AdaIR(ICLR2025)"),
    "LUCIDMine": ("LUCIDMine", "LUCID", "LUCIDMine(v2)"),
}

METHODS = ["Input", "DCP", "CLAHE", "Retinex", "AdaIR", "LUCIDMine"]

METRICS = [
    ("PSNR ↑", "PSNR", True, "dB"),
    ("SSIM ↑", "SSIM", True, ""),
    ("MAE ↓", "MAE", False, ""),
    ("Vis ↑", "Vis", True, ""),
]


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "mathtext.fontset": "stix",
        }
    )


def load_data() -> dict:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing comparison data. Expected {DATA_PATH} or {FALLBACK_DATA_PATH}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_method(methods: dict, canonical: str) -> dict:
    aliases = METHOD_ALIASES[canonical]
    for alias in aliases:
        if alias in methods:
            return methods[alias]
    lower = {str(k).lower(): v for k, v in methods.items()}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise KeyError(f"Cannot find method {canonical}; tried aliases {aliases}.")


def collect_table(data: dict) -> list[dict[str, float | str | int]]:
    method_data = data["methods"]
    rows = []
    for method in METHODS:
        record = resolve_method(method_data, method)
        rows.append(
            {
                "Method": method,
                "n": int(record.get("n", 0)),
                "PSNR": float(record["psnr"]),
                "SSIM": float(record["ssim"]),
                "MAE": float(record["mae"]),
                "Vis": float(record["vis"]),
            }
        )
    return rows


def save_tables(rows: list[dict[str, float | str | int]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "comparison_lowres_448x256_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "n", "PSNR", "SSIM", "MAE", "Vis"])
        writer.writeheader()
        writer.writerows(rows)

    tex_path = TABLE_DIR / "comparison_lowres_448x256_summary.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative comparison at 448$\times$256 resolution. Higher PSNR, SSIM, and Vis are better; lower MAE is better.}",
        r"\label{tab:comparison_lowres_448x256}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & $n$ & PSNR$\uparrow$ & SSIM$\uparrow$ & MAE$\downarrow$ & Vis$\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Method']} & {row['n']} & {row['PSNR']:.2f} & {row['SSIM']:.3f} & "
            f"{row['MAE']:.3f} & {row['Vis']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


def normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def draw_arc_panel(ax: plt.Axes, rows: list[dict[str, float | str | int]], title: str, key: str, unit: str) -> None:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    norm = normalize(values)

    theta_start = math.radians(135.0)
    max_span = math.radians(245.0)
    theta_bg = np.linspace(theta_start, theta_start - max_span, 300)

    radii = np.linspace(1.04, 0.46, len(METHODS))
    cmap = plt.get_cmap("Blues")
    colors = [cmap(v) for v in np.linspace(0.48, 0.86, len(METHODS))]

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.30, 1.14)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    for idx, (method, radius, val, score, color) in enumerate(zip(METHODS, radii, values, norm, colors)):
        ax.plot(theta_bg, np.full_like(theta_bg, radius), color="#E8EEF7", lw=8.0, solid_capstyle="round", zorder=1)
        theta = np.linspace(theta_start, theta_start - max_span * (0.08 + 0.92 * score), 220)
        ax.plot(theta, np.full_like(theta, radius), color=color, lw=8.0, solid_capstyle="round", zorder=2)

    # Method names are placed as a compact left-side radial list rather than
    # on the arcs themselves; this prevents label collision in four-panel layout.
    label_ys = np.linspace(0.62, 0.34, len(METHODS))
    for method, y, color in zip(METHODS, label_ys, colors):
        ax.text(
            0.035,
            y,
            method,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.8,
            color="#253041",
        )
        ax.plot(
            [0.005, 0.027],
            [y, y],
            transform=ax.transAxes,
            color=color,
            lw=3.0,
            solid_capstyle="round",
            clip_on=False,
        )

    ax.text(
        0.5,
        -0.105,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1F2937",
    )


def save_all(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = FIG_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")


def main() -> None:
    data = load_data()
    rows = collect_table(data)
    save_tables(rows)
    configure_style()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11.2, 2.65),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    for ax, (title, metric_key, _higher_is_better, unit) in zip(axes, METRICS):
        draw_arc_panel(ax, rows, title, metric_key, unit)

    fig.subplots_adjust(left=0.012, right=0.995, top=0.965, bottom=0.185, wspace=0.18)
    save_all(fig)


if __name__ == "__main__":
    main()
