from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULLREF_PATH = ROOT / "figure_data" / "metrics" / "test_fullref_metrics.csv"
ADAIR_PATH = ROOT / "figure_data" / "metrics" / "adair_eval.json"
VIS_PATH = ROOT / "figure_data" / "metrics" / "vis_metrics_all_methods_with_adair.csv"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "figure_data" / "tables"
OUT_STEM = "fig2_quantitative_arc_profile"

FULLREF_ALIASES = {
    "Input": ("input", "Input"),
    "DCP": ("dcp", "DCP"),
    "CLAHE": ("clahe", "CLAHE"),
    "Retinex": ("retinex", "Retinex"),
    "LUCIDMine": ("lucidmine_modal_v2", "LUCIDMine", "lucidmine", "LUCID"),
}

VIS_ALIASES = {
    "Input": ("input", "Input"),
    "DCP": ("dcp", "DCP"),
    "CLAHE": ("clahe", "CLAHE"),
    "Retinex": ("retinex", "Retinex"),
    "AdaIR": ("adair", "AdaIR", "AdaIR(ICLR2025)"),
    "LUCIDMine": ("lucidmine", "LUCIDMine", "lucidmine_modal_v2", "LUCID"),
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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")


def read_fullref_rows() -> dict[str, dict[str, str]]:
    require_file(FULLREF_PATH)
    rows = list(csv.DictReader(FULLREF_PATH.read_text(encoding="utf-8-sig").splitlines()))
    return {str(row["method"]).lower(): row for row in rows}


def read_adair_eval() -> dict:
    require_file(ADAIR_PATH)
    return json.loads(ADAIR_PATH.read_text(encoding="utf-8"))


def read_vis_means() -> dict[str, float]:
    require_file(VIS_PATH)
    rows = list(csv.DictReader(VIS_PATH.read_text(encoding="utf-8-sig").splitlines()))
    grouped: dict[str, list[float]] = {}
    for row in rows:
        method = str(row["method"]).lower()
        grouped.setdefault(method, []).append(float(row["vis"]))
    return {method: statistics.mean(values) for method, values in grouped.items()}


def resolve_row(rows: dict[str, dict[str, str]], canonical: str) -> dict[str, str]:
    for alias in FULLREF_ALIASES[canonical]:
        row = rows.get(alias.lower())
        if row is not None:
            return row
    raise KeyError(f"Cannot find full-reference row for {canonical}; tried {FULLREF_ALIASES[canonical]}")


def resolve_vis(vis_means: dict[str, float], canonical: str) -> float:
    for alias in VIS_ALIASES[canonical]:
        if alias.lower() in vis_means:
            return vis_means[alias.lower()]
    raise KeyError(f"Cannot find Vis rows for {canonical}; tried {VIS_ALIASES[canonical]}")


def collect_table() -> list[dict[str, float | str | int]]:
    fullref_rows = read_fullref_rows()
    adair = read_adair_eval()
    vis_means = read_vis_means()
    rows = []
    for method in METHODS:
        if method == "AdaIR":
            n = int(adair.get("n", 0))
            psnr = float(adair["full_psnr"])
            ssim = float(adair["full_ssim"])
            mae = float(adair["masked_l1"])
        else:
            record = resolve_row(fullref_rows, method)
            n = int(record.get("n", 0))
            psnr = float(record["psnr"])
            ssim = float(record["ssim"])
            mae = float(record["mae"])
        rows.append(
            {
                "Method": method,
                "n": n,
                "PSNR": psnr,
                "SSIM": ssim,
                "MAE": mae,
                "Vis": resolve_vis(vis_means, method),
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
    rows = collect_table()
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
