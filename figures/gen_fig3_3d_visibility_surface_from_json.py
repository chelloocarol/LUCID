from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "per_scene_vis_summary.json"
FIG_DIR = ROOT / "figures"
OUT_STEM = "fig3_3d_visibility_surface"

METHOD_RENAME = {
    "RIDCP": "AdaIR",
    "LUCID": "LUCIDMine",
}

METHOD_ORDER = ["Input", "DCP", "CLAHE", "Retinex", "AdaIR", "LUCIDMine"]


def normalize_method(name: str) -> str:
    return METHOD_RENAME.get(str(name), str(name))


def pick_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key in record:
            return record[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    raise KeyError(f"Missing any of keys {keys} in record: {record}")


def flatten_records(data: Any) -> list[dict[str, Any]]:
    """Accept common JSON layouts and return scene/method/Vis records."""
    records: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if any(k.lower() == "method" for k in item) and any(k.lower() in {"vis", "visibility"} for k in item):
                scene = pick_value(item, ("scene", "Scene", "image", "name"))
                method = pick_value(item, ("method", "Method"))
                vis = pick_value(item, ("Vis", "vis", "visibility", "Visibility"))
                records.append({"scene": str(scene), "method": normalize_method(str(method)), "Vis": float(vis)})
            else:
                scene = pick_value(item, ("scene", "Scene", "image", "name"))
                for key, value in item.items():
                    if str(key).lower() in {"scene", "image", "name"}:
                        continue
                    if isinstance(value, (int, float)):
                        records.append({"scene": str(scene), "method": normalize_method(str(key)), "Vis": float(value)})
        return records

    if isinstance(data, dict):
        if {"scenes", "methods"}.issubset(data.keys()) and any(k in data for k in ("values", "Vis", "vis", "visibility")):
            scenes = [str(s) for s in data["scenes"]]
            methods = [normalize_method(str(m)) for m in data["methods"]]
            values = data.get("values", data.get("Vis", data.get("vis", data.get("visibility"))))
            for i, scene in enumerate(scenes):
                for j, method in enumerate(methods):
                    records.append({"scene": scene, "method": method, "Vis": float(values[i][j])})
            return records

        # Nested layout: {scene: {method: Vis, ...}, ...}
        for scene, methods in data.items():
            if not isinstance(methods, dict):
                continue
            for method, vis in methods.items():
                if isinstance(vis, dict):
                    vis = pick_value(vis, ("Vis", "vis", "visibility", "Visibility"))
                records.append({"scene": str(scene), "method": normalize_method(str(method)), "Vis": float(vis)})
        return records

    raise TypeError(f"Unsupported JSON root type: {type(data)!r}")


def ordered_unique(values: list[str], preferred: list[str] | None = None) -> list[str]:
    seen = set()
    result: list[str] = []
    if preferred:
        for value in preferred:
            if value in values and value not in seen:
                result.append(value)
                seen.add(value)
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "mathtext.fontset": "stix",
        }
    )


def save_all(fig: plt.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = FIG_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing data file: {DATA_PATH}\n"
            "Expected a per-scene visibility summary JSON with fields scene, method, and Vis."
        )

    records = flatten_records(json.loads(DATA_PATH.read_text(encoding="utf-8")))
    if not records:
        raise ValueError(f"No plottable records found in {DATA_PATH}")

    scenes = ordered_unique([r["scene"] for r in records])
    methods = ordered_unique([r["method"] for r in records], METHOD_ORDER)

    z = np.full((len(scenes), len(methods)), np.nan, dtype=float)
    for record in records:
        i = scenes.index(record["scene"])
        j = methods.index(record["method"])
        z[i, j] = record["Vis"]

    if np.isnan(z).any():
        missing = [
            f"{scene}/{method}"
            for i, scene in enumerate(scenes)
            for j, method in enumerate(methods)
            if np.isnan(z[i, j])
        ]
        raise ValueError("Missing Vis values for: " + ", ".join(missing[:20]))

    configure_style()

    x, y = np.meshgrid(np.arange(len(methods)), np.arange(len(scenes)))
    fig = plt.figure(figsize=(6.2, 3.55))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x,
        y,
        z,
        cmap="viridis",
        linewidth=0.25,
        edgecolor="white",
        alpha=0.96,
        antialiased=True,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("Vis", labelpad=4)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(scenes)))
    ax.set_yticklabels(scenes, rotation=-8)
    ax.tick_params(axis="z", labelsize=8, pad=1)
    ax.view_init(elev=25, azim=-55)
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.02)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.035)
    cbar.set_label("Visibility score (Vis)", rotation=270, labelpad=12)

    save_all(fig)


if __name__ == "__main__":
    main()
