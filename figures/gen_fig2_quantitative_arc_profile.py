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

METRICS = [
    {
        "title": "PSNR ↑",
        "key": "psnr",
        "table_key": "PSNR",
        "ticks": [0, 5, 10, 15, 20, 25],
        "rmax": 25.0,
    },
    {
        "title": "SSIM ↑",
        "key": "ssim",
        "table_key": "SSIM",
        "ticks": [0, 0.25, 0.50, 0.75, 1.00],
        "rmax": 1.0,
    },
    {
        "title": "MAE ↓",
        "key": "mae",
        "table_key": "MAE",
        "ticks": [0, 0.05, 0.10, 0.15, 0.20, 0.25],
        "rmax": 0.25,
    },
    {
        "title": "Vis ↑",
        "key": "vis",
        "table_key": "Vis",
        "ticks": [0, 0.25, 0.50, 0.75, 1.00],
        "rmax": 1.0,
    },
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


def load_rows() -> list[dict[str, float | str | int]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing arc-chart data file: {DATA_PATH}")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    methods = data["methods_order"]
    metrics = data["metrics"]

    rows: list[dict[str, float | str | int]] = []
    for method in methods:
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


def tick_labels(ticks: list[float]) -> list[str]:
    labels = []
    for tick in ticks:
        if tick >= 1 and abs(tick - round(tick)) < 1e-8:
            labels.append(str(int(round(tick))))
        elif tick == 0:
            labels.append("0")
        else:
            labels.append(f"{tick:.2f}".rstrip("0").rstrip("."))
    return labels


def draw_arc_panel(
    ax: plt.Axes,
    rows: list[dict[str, float | str | int]],
    metric: dict[str, object],
) -> None:
    methods = [str(row["Method"]) for row in rows]
    values = np.array([float(row[str(metric["table_key"])]) for row in rows], dtype=float)
    rmax = float(metric["rmax"])
    ticks = [float(t) for t in metric["ticks"]]  # type: ignore[index]

    theta_start = math.radians(135.0)
    max_span = math.radians(245.0)
    theta_ref = np.linspace(theta_start, theta_start - max_span, 360)

    cmap = plt.get_cmap("Blues")
    colors = [cmap(v) for v in np.linspace(0.46, 0.88, len(methods))]

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, rmax)
    ax.set_facecolor("white")
    ax.spines["polar"].set_visible(False)

    # Hide angular coordinates; keep only subtle radial value ticks.
    ax.set_xticks([])
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels(ticks), fontsize=6.5, color="#7B8494")
    ax.set_rlabel_position(112)
    ax.yaxis.grid(True, color="#E4EAF2", linewidth=0.55)
    ax.xaxis.grid(False)

    # Method-specific arcs. Radius and arc length both follow the metric value;
    # faint full-span arcs provide a visual reference at the same radius.
    for method, value, color in zip(methods, values, colors):
        clipped = min(max(value, 0.0), rmax)
        score = clipped / rmax if rmax else 0.0
        ax.plot(
            theta_ref,
            np.full_like(theta_ref, clipped),
            color="#ECF1F8",
            lw=8.0,
            solid_capstyle="round",
            zorder=1,
        )
        theta = np.linspace(theta_start, theta_start - max_span * score, 260)
        ax.plot(
            theta,
            np.full_like(theta, clipped),
            color=color,
            lw=8.0,
            solid_capstyle="round",
            zorder=3,
        )

    # Left-side method list, consistent across panels.
    label_ys = np.linspace(0.62, 0.34, len(methods))
    for method, y, color in zip(methods, label_ys, colors):
        ax.text(
            0.035,
            y,
            method,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.7,
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
        -0.11,
        str(metric["title"]),
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
    configure_style()
    rows = load_rows()
    save_tables(rows)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11.2, 2.75),
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
