"""
After LUCIDMine training completes, update all result tables.

Usage:
    python tools/update_results_after_training.py \
        --infer_dir experiment/infer_test/lucidmine_modal \
        --vis_json  experiment/eval/vis_summary_modal.json \
        --run_name  lucidmine_modal \
        --exp_id    EXP-201

The script:
  1. Reads PSNR/SSIM/MAE from test_fullref_metrics.csv (expected to have new row already)
  2. Reads Vis from vis_summary JSON
  3. Appends rows to reproduction_results.csv
  4. Updates experiment/eval/summary_40video_test.md
"""
import argparse
import csv
import json
import os
import sys


def load_fullref(csv_path, method_name):
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["method"] == method_name:
                return {
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                    "mae":  float(row["mae"]),
                    "n":    int(row["n"]),
                }
    return None


def load_vis(json_path, method_name):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if method_name in data:
        return data[method_name].get("vis", None)
    return None


def append_to_reproduction_csv(csv_path, exp_id, run_name, metrics):
    rows_to_add = []
    for metric_name, val in metrics.items():
        rows_to_add.append({
            "实验编号": exp_id,
            "实验名称": "LUCIDMine微调后评估",
            "模型": f"LUCIDMine({run_name})",
            "数据集": "lucidmine-40-video test set",
            "n": 152,
            "指标": metric_name,
            "值": f"{val:.6f}" if isinstance(val, float) else str(val),
            "标准差": "N/A",
            "来源": "experiment/eval/test_fullref_metrics.csv",
            "日期": "2026-06-02",
            "备注": "Modal A10G 100epoch 训练" if "modal" in run_name else "CPU 100epoch 训练",
        })

    # Check existing rows in reproduction_results.csv
    existing = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            existing.add((row["实验编号"], row["指标"]))

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        fieldnames = list(csv.DictReader(f).fieldnames)

    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows_to_add:
            key = (row["实验编号"], row["指标"])
            if key not in existing:
                writer.writerow(row)
                print(f"  + Added {key}")
            else:
                print(f"  ~ Skipped (exists): {key}")


def update_summary_md(md_path, run_name, metrics):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    # Replace the pending row for 100epoch LUCIDMine
    old = "| **LUCIDMine (100epoch 微调后)** | **待训练** | **待训练** | **待训练** | ⏳ 等待 Modal GPU (token_secret) |"
    psnr = metrics.get("PSNR/dB", "N/A")
    ssim = metrics.get("SSIM", "N/A")
    mae  = metrics.get("MAE", "N/A")
    label = "Modal A10G" if "modal" in run_name else "CPU 100ep"
    new  = f"| **LUCIDMine (100epoch, {label})** | **{psnr:.3f}** | **{ssim:.4f}** | **{mae:.4f}** | ✅ 已完成 |"
    if old in content:
        content = content.replace(old, new)
        print(f"  Updated summary table: PSNR={psnr:.3f}")

    # Replace Vis pending row
    old_vis = "| **LUCIDMine (微调后)** | ⏳ |"
    vis = metrics.get("Vis", None)
    if vis is not None:
        new_vis = f"| **LUCIDMine (微调后, {label})** | **{vis:.3f}** |"
        if old_vis in content:
            content = content.replace(old_vis, new_vis)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, help="e.g. lucidmine_modal or lucidmine_cpu100ep")
    parser.add_argument("--exp_id", default="EXP-201", help="Experiment ID for reproduction_results.csv")
    parser.add_argument("--vis_json", default=None, help="Path to vis_summary JSON with the run_name key")
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    fullref_csv = os.path.join(ROOT, "experiment/eval/test_fullref_metrics.csv")
    repro_csv   = os.path.join(ROOT, "reproduction_results.csv")
    summary_md  = os.path.join(ROOT, "experiment/eval/summary_40video_test.md")

    metrics = {}

    fr = load_fullref(fullref_csv, args.run_name)
    if fr:
        metrics["PSNR/dB"] = fr["psnr"]
        metrics["SSIM"]    = fr["ssim"]
        metrics["MAE"]     = fr["mae"]
        print(f"Full-ref: PSNR={fr['psnr']:.4f} SSIM={fr['ssim']:.4f} MAE={fr['mae']:.4f}")
    else:
        print(f"WARNING: {args.run_name} not found in {fullref_csv} — run eval_after_training.sh first")

    if args.vis_json:
        vis = load_vis(args.vis_json, args.run_name)
        if vis is not None:
            metrics["Vis"] = vis
            print(f"Vis: {vis:.4f}")

    if not metrics:
        print("No metrics found — nothing to update.")
        sys.exit(1)

    print("\nAppending to reproduction_results.csv ...")
    append_to_reproduction_csv(repro_csv, args.exp_id, args.run_name, metrics)

    print("Updating summary_40video_test.md ...")
    update_summary_md(summary_md, args.run_name, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
