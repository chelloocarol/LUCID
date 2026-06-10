from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "figure_data" / "metrics" / "arc_chart_data.json"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "figure_data" / "tables"
OUT_STEM = "fig2_quantitative_arc_profile"

DISPLAY_METHODS = ["LUCIDMine", "AdaIR", "CLAHE", "Input", "DCP", "Retinex"]

METRICS = [
    {"title": "PSNR ↑", "key": "psnr", "table_key": "PSNR", "rmax": 25.0, "higher": True, "fmt": "{:.2f}"},
    {"title": "SSIM ↑", "key": "ssim", "table_key": "SSIM", "rmax": 1.0, "higher": True, "fmt": "{:.3f}"},
    {"title": "MAE ↓", "key": "mae", "table_key": "MAE", "rmax": 0.25, "higher": False, "fmt": "{:.3f}"},
    {"title": "Vis ↑", "key": "vis", "table_key": "Vis", "rmax": 1.0, "higher": True, "fmt": "{:.3f}"},
]


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "mathtext.fontset": "stix",
        }
    )


def load_rows() -> list[dict[str, float | str | int]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing arc-chart data file: {DATA_PATH}")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    metrics = data["metrics"]

    rows: list[dict[str, float | str | int]] = []
    for method in data["methods_order"]:
        record = metrics[method]
        rows.append(
            {
                "Method": method,
                "n": int(data.get("n", 152)),
                "PSNR": float(record["psnr"]),
                "SSIM": float(record["ssim"]),
                "MAE": float(record["mae"]),
                "Vis": float(record["vis"]),
            }
        )
    return rows


def save_tables(rows: list[dict[str, float | str | int]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "arc_chart_data_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "n", "PSNR", "SSIM", "MAE", "Vis"])
        writer.writeheader()
        writer.writerows(rows)

    tex_path = TABLE_DIR / "arc_chart_data_summary.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Unified 448$\times$256 quantitative comparison. Higher PSNR, SSIM, and Vis are better; lower MAE is better.}",
        r"\label{tab:arc_chart_data_summary}",
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


def row_by_method(rows: list[dict[str, float | str | int]]) -> dict[str, dict[str, float | str | int]]:
    return {str(row["Method"]): row for row in rows}


def quality_score(value: float, rmax: float, higher_is_better: bool) -> float:
    clipped = min(max(value, 0.0), rmax)
    raw = clipped / rmax if rmax else 0.0
    return raw if higher_is_better else 1.0 - raw


def draw_arc_panel(ax: plt.Axes, rows: list[dict[str, float | str | int]], metric: dict[str, object]) -> None:
    lookup = row_by_method(rows)
    rmax = float(metric["rmax"])
    higher = bool(metric["higher"])
    key = str(metric["table_key"])

    theta_start = math.radians(135.0)
    max_span = math.radians(245.0)
    theta_bg = np.linspace(theta_start, theta_start - max_span, 360)
    method_radii = np.linspace(1.04, 0.46, len(DISPLAY_METHODS))

    cmap = plt.get_cmap("Blues")
    colors = [cmap(v) for v in np.linspace(0.88, 0.44, len(DISPLAY_METHODS))]

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.30, 1.16)
    ax.set_facecolor("white")
    ax.spines["polar"].set_visible(False)

    # No coordinate ticks: the arcs are categorical method layers, not radial-value axes.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    lucid_tail = None
    lucid_value = None
    lucid_radius = None

    for method, radius, color in zip(DISPLAY_METHODS, method_radii, colors):
        value = float(lookup[method][key])
        score = quality_score(value, rmax, higher)
        span = max_span * (0.06 + 0.94 * score)

        ax.plot(
            theta_bg,
            np.full_like(theta_bg, radius),
            color="#ECF1F8",
            lw=8.5,
            solid_capstyle="round",
            zorder=1,
        )

        theta = np.linspace(theta_start, theta_start - span, 260)
        ax.plot(
            theta,
            np.full_like(theta, radius),
            color=color,
            lw=8.5,
            solid_capstyle="round",
            zorder=3,
        )

        if method == "LUCIDMine":
            lucid_tail = theta[-1]
            lucid_value = value
            lucid_radius = radius

    # Left-side method list.
    label_ys = np.linspace(0.62, 0.34, len(DISPLAY_METHODS))
    for method, y, color in zip(DISPLAY_METHODS, label_ys, colors):
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

    if lucid_tail is not None and lucid_radius is not None and lucid_value is not None:
        label = str(metric["fmt"]).format(lucid_value)
        ax.text(
            lucid_tail,
            lucid_radius + 0.035,
            label,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#1F2937",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=5,
        )

    ax.text(
        0.5,
        -0.11,
        str(metric["title"]),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
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
    configure_style()
    rows = load_rows()
    save_tables(rows)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11.2, 2.65),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, METRICS):
        draw_arc_panel(ax, rows, metric)

    fig.subplots_adjust(left=0.012, right=0.995, top=0.965, bottom=0.19, wspace=0.20)
    save_all(fig)


if __name__ == "__main__":
    main()
