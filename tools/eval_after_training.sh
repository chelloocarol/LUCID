#!/usr/bin/env bash
# Post-training evaluation pipeline for LUCIDMine.
#
# Usage:
#   bash tools/eval_after_training.sh <checkpoint_path> <run_name>
#
# Example (CPU run):
#   bash tools/eval_after_training.sh \
#       experiment/LUCIDMine/cpu_run_100ep/best.pth \
#       lucidmine_cpu100ep
#
# Steps:
#   1. Infer test set (n=152) with the checkpoint
#   2. Compute full-reference PSNR/SSIM/MAE vs RIDCP targets
#   3. Compute Vis composite metric (joint normalisation with all baselines)
#   4. Print comparison table

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CKPT="${1:?Usage: $0 <checkpoint_path> <run_name>}"
RUN="${2:?Usage: $0 <checkpoint_path> <run_name>}"

INFER_DIR="experiment/infer_test/${RUN}"
REF_DIR="/home/user/lucidmine-40-video-dataset/data/test/target"
INPUT_DIR="experiment/infer_test/input"

echo "=== Step 1: Infer test set with ${CKPT} ==="
python3 tools/infer_folder_ckpt.py \
  --checkpoint_path "${CKPT}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${INFER_DIR}" \
  --model_arch lucidmine \
  --state_key model

echo "=== Step 2: Full-reference metrics ==="
python3 - <<PYEOF
import csv, glob, os, json
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn

infer_dir = "${INFER_DIR}"
manifest  = "data/mine_manifest.csv"
ref_base  = "${REF_DIR}"

with open(manifest, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
test_rows = {r['sample_id']: r for r in rows if r['split']=='test'}

psnrs, ssims, maes = [], [], []
for png in sorted(glob.glob(os.path.join(infer_dir, "*.png"))):
    stem = os.path.splitext(os.path.basename(png))[0]
    if stem not in test_rows:
        continue
    ref_path = test_rows[stem]['target_path']
    if not os.path.exists(ref_path):
        continue
    pred = np.array(Image.open(png).convert("RGB")).astype(np.float32)/255.0
    ref  = np.array(Image.open(ref_path).convert("RGB").resize(
               (pred.shape[1],pred.shape[0]),Image.BICUBIC)).astype(np.float32)/255.0
    psnrs.append(psnr_fn(ref, pred, data_range=1.0))
    ssims.append(ssim_fn(ref, pred, data_range=1.0, channel_axis=2))
    maes.append(float(np.abs(ref-pred).mean()))

result = dict(method="${RUN}", psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)),
              mae=float(np.mean(maes)), n=len(psnrs))
print(f"PSNR={result['psnr']:.4f}  SSIM={result['ssim']:.4f}  MAE={result['mae']:.4f}  n={result['n']}")

# Append to test_fullref_metrics.csv
out_csv = "experiment/eval/test_fullref_metrics.csv"
with open(out_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['method','psnr','ssim','mae','n'])
    writer.writerow(result)
print(f"Appended to {out_csv}")
PYEOF

echo "=== Step 3: Vis metric (joint normalisation with baselines) ==="
python3 tools/eval_vis_metric.py \
  --dirs input=experiment/infer_test/input \
         clahe=experiment/infer_test/clahe \
         retinex=experiment/infer_test/retinex \
         dcp=experiment/infer_test/dcp \
         student=experiment/infer_test/student \
         lucidmine_init=experiment/infer_test/lucidmine_init \
         "${RUN}=${INFER_DIR}" \
  --ref_dir "${REF_DIR}" \
  --output_csv "experiment/eval/vis_metrics_${RUN}.csv" \
  --summary_json "experiment/eval/vis_summary_${RUN}.json"

echo ""
echo "=== Done. Results written to ==="
echo "  experiment/eval/test_fullref_metrics.csv"
echo "  experiment/eval/vis_metrics_${RUN}.csv"
echo "  experiment/eval/vis_summary_${RUN}.json"
