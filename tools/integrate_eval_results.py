"""
Read eval_mine_per_video.py JSON output and update all audit tables.

Usage:
    python tools/integrate_eval_results.py \
        --eval_json experiment/eval/lucidmine_v2_eval.json \
        --method_name lucidmine_modal_v2 \
        --exp_id EXP-201
"""
import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_eval_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def get_overall_metrics(data):
    o = data["overall"]
    return {
        "psnr":  o["full_psnr"],
        "ssim":  o["full_ssim"],
        "mae":   o["masked_l1"],
        "n": 152,
    }


def update_test_fullref_csv(csv_path, method_name, metrics):
    rows = []
    updated = False
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["method"] == method_name:
                row.update({"psnr": str(metrics["psnr"]), "ssim": str(metrics["ssim"]),
                             "mae": str(metrics["mae"]), "n": str(metrics["n"])})
                updated = True
            rows.append(row)

    if not updated:
        rows.append({"method": method_name, "psnr": str(metrics["psnr"]),
                     "ssim": str(metrics["ssim"]), "mae": str(metrics["mae"]),
                     "n": str(metrics["n"])})

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {'Updated' if updated else 'Added'} {method_name} in {csv_path}")


def update_reproduction_csv(csv_path, exp_id, method_name, metrics):
    existing_keys = set()
    rows = []
    fieldnames = None
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_keys.add((row["实验编号"], row["指标"]))
            rows.append(row)

    new_rows = [
        ("PSNR/dB", f"{metrics['psnr']:.6f}"),
        ("SSIM",    f"{metrics['ssim']:.6f}"),
        ("MAE",     f"{metrics['mae']:.6f}"),
    ]
    added = 0
    for metric_name, val in new_rows:
        key = (exp_id, metric_name)
        if key not in existing_keys:
            rows.append({
                "实验编号": exp_id,
                "实验名称": "LUCIDMine微调后评估",
                "模型": f"LUCIDMine({method_name})",
                "数据集": "lucidmine-40-video test set",
                "n": "152",
                "指标": metric_name,
                "值": val,
                "标准差": "N/A",
                "来源": "experiment/eval/lucidmine_v2_eval.json",
                "日期": "2026-06-02",
                "备注": "Modal A10G，83 epoch（best@ep48）；代理集RIDCP伪标签参考",
            })
            added += 1
        else:
            # Update existing placeholder rows
            for row in rows:
                if row["实验编号"] == exp_id and row["指标"] == metric_name:
                    row["值"] = val
                    row["来源"] = "experiment/eval/lucidmine_v2_eval.json"
                    row["备注"] = "Modal A10G，83 epoch（best@ep48）；代理集RIDCP伪标签参考"
                    added += 1
                    break

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Updated reproduction_results.csv ({added} rows for {exp_id})")


def update_summary_md(md_path, method_name, metrics):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    psnr = metrics["psnr"]
    ssim = metrics["ssim"]
    mae  = metrics["mae"]

    old = "| **LUCIDMine (100epoch 微调后)** | **待训练** | **待训练** | **待训练** | ⏳ 等待 Modal GPU (token_secret) |"
    new = (f"| **LUCIDMine (Modal A10G, ep48)** | "
           f"**{psnr:.3f}** | **{ssim:.4f}** | **{mae:.4f}** | ✅ EXP-201 完成 |")
    if old in content:
        content = content.replace(old, new)
        print(f"  Updated LUCIDMine row in summary_40video_test.md")
    else:
        print(f"  WARNING: pending placeholder not found in summary_40video_test.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def update_metric_consistency_csv(csv_path, metrics):
    rows = []
    updated = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["论文章节"] == "§4.2" and row["方法名称"] == "LUCIDMine":
                claimed = float(row["论文声明值"])
                if row["指标名称"] == "PSNR↑/dB":
                    actual = metrics["psnr"]
                    diff = actual - claimed
                    rel = f"{diff/claimed*100:+.1f}%"
                    row["复现实验值"] = f"{actual:.3f}(代理集)"
                    row["差值"] = f"{diff:+.3f}"
                    row["相对误差"] = rel
                    row["是否一致"] = "不可比较（测试集不同）"
                    row["修改建议"] = (
                        f"代理集(n=152,RIDCP伪标签)PSNR={actual:.3f}，与论文(n=288,真实配对)"
                        f"差{diff:+.2f}dB；测试集不等价，差距主要来自数据集差异"
                    )
                    updated += 1
                elif row["指标名称"] == "SSIM↑":
                    actual = metrics["ssim"]
                    diff = actual - claimed
                    row["复现实验值"] = f"{actual:.4f}(代理集)"
                    row["差值"] = f"{diff:+.4f}"
                    row["相对误差"] = f"{diff/claimed*100:+.1f}%"
                    row["是否一致"] = "不可比较（测试集不同）"
                    row["修改建议"] = (
                        f"代理集SSIM={actual:.4f}，与论文差{diff:+.4f}；测试集不等价"
                    )
                    updated += 1
                elif row["指标名称"] == "MAE↓":
                    actual = metrics["mae"]
                    diff = actual - claimed
                    row["复现实验值"] = f"{actual:.4f}(代理集)"
                    row["差值"] = f"{diff:+.4f}"
                    row["相对误差"] = f"{diff/claimed*100:+.1f}%"
                    row["是否一致"] = "不可比较（测试集不同）"
                    row["修改建议"] = (
                        f"代理集MAE={actual:.4f}（masked L1），与论文差{diff:+.4f}；测试集不等价"
                    )
                    updated += 1
            rows.append(row)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Updated {updated} LUCIDMine rows in metric_consistency_table.csv")


def update_paper_claims_audit(md_path, metrics):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    psnr = metrics["psnr"]
    ssim = metrics["ssim"]
    mae  = metrics["mae"]

    old = "| **LUCIDMine** | **23.42** | **待EXP-201评估** | **0.956** | **待EXP-201评估** | **0.931** | **待EXP-201评估** | ⏳ 权重已就绪(ep48 val_psnr=23.14)，正式评估运行中 |"
    new = (f"| **LUCIDMine** | **23.42** | **{psnr:.3f}(代理集)** | "
           f"**0.956** | **{ssim:.4f}(代理集)** | "
           f"**0.931** | **待Vis计算** | "
           f"✅ EXP-201完成(代理集n=152,RIDCP伪标签) |")
    if old in content:
        content = content.replace(old, new)
        print(f"  Updated LUCIDMine row in paper_claims_audit.md")
    else:
        print(f"  WARNING: LUCIDMine row pattern not found in paper_claims_audit.md — manual update needed")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def print_summary(metrics):
    print("\n" + "="*60)
    print("EXP-201 LUCIDMine 评估结果（代理集，n=152，RIDCP伪标签）")
    print("="*60)
    print(f"  full_psnr  : {metrics['psnr']:.4f} dB  (论文声明: 23.42 dB)")
    print(f"  full_ssim  : {metrics['ssim']:.4f}      (论文声明: 0.956)")
    print(f"  masked_mae : {metrics['mae']:.4f}      (论文声明: 0.057)")
    print(f"  n          : {metrics['n']}")
    print()
    print("⚠️  注意: 代理集(RIDCP伪标签)≠论文测试集(真实配对)，绝对值不可直接比较")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", default="experiment/eval/lucidmine_v2_eval.json")
    parser.add_argument("--method_name", default="lucidmine_modal_v2")
    parser.add_argument("--exp_id", default="EXP-201")
    args = parser.parse_args()

    eval_path = os.path.join(ROOT, args.eval_json)
    if not os.path.exists(eval_path):
        print(f"ERROR: {eval_path} not found. Run eval_mine_per_video.py first.")
        sys.exit(1)

    data = load_eval_json(eval_path)
    metrics = get_overall_metrics(data)
    print_summary(metrics)

    print("\nUpdating tables...")
    update_test_fullref_csv(
        os.path.join(ROOT, "experiment/eval/test_fullref_metrics.csv"),
        args.method_name, metrics)
    update_reproduction_csv(
        os.path.join(ROOT, "reproduction_results.csv"),
        args.exp_id, args.method_name, metrics)
    update_summary_md(
        os.path.join(ROOT, "experiment/eval/summary_40video_test.md"),
        args.method_name, metrics)
    update_metric_consistency_csv(
        os.path.join(ROOT, "metric_consistency_table.csv"),
        metrics)
    update_paper_claims_audit(
        os.path.join(ROOT, "paper_claims_audit.md"),
        metrics)
    print("\n✅ All tables updated.")


if __name__ == "__main__":
    main()
