#!/bin/bash
# Finalize AdaIR evaluation after all 152 images are inferred.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COUNT=$(ls experiment/infer_test/adair_dehaze/ | wc -l)
if [ "$COUNT" -lt 152 ]; then
    echo "ERROR: Only $COUNT/152 AdaIR images found. Wait for inference to complete."
    exit 1
fi
echo "AdaIR: $COUNT/152 images found. Proceeding..."

echo ""
echo "=== Step 1: AdaIR PSNR/SSIM/MAE at 448x256 ==="
python3 tools/eval_adair_metrics.py \
    --pred_dir experiment/infer_test/adair_dehaze \
    --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
    --eval_size 448 256 \
    --output_json experiment/eval/adair_eval.json

echo ""
echo "=== Step 2: Update lowres comparison table with AdaIR results ==="
python3 - << 'PYEOF'
import json, os

ROOT = os.getcwd()

# Load existing comparison table
cmp_path = os.path.join(ROOT, "experiment/eval/comparison_lowres_448x256.json")
with open(cmp_path) as f:
    cmp = json.load(f)

# Load AdaIR results
adair_path = os.path.join(ROOT, "experiment/eval/adair_eval.json")
with open(adair_path) as f:
    adair = json.load(f)

# Update AdaIR entry
cmp["methods"]["AdaIR(ICLR2025)"] = {
    "n": adair["n"],
    "psnr": adair["full_psnr"],
    "ssim": adair["full_ssim"],
    "mae": adair["masked_l1"],
    "psnr_std": adair["psnr_std"],
    "vis": None,  # computed below
}

# Compute Vis for AdaIR at 448x256 using the same normalization stats
# Load all method records to normalize consistently
# For now, mark Vis as pending - will be computed separately
cmp["note"] = (
    "All methods evaluated at 448x256 for fair comparison. "
    "Outputs resized from native resolution. Targets resized from 1920x1088. "
    "NOT directly comparable to full-resolution tables."
)

with open(cmp_path, "w") as f:
    json.dump(cmp, f, indent=2, ensure_ascii=False)

print("Updated comparison_lowres_448x256.json with final AdaIR results")
print(f"AdaIR (n={adair['n']}): PSNR={adair['full_psnr']:.4f}, SSIM={adair['full_ssim']:.4f}, MAE={adair['masked_l1']:.4f}")
PYEOF

echo ""
echo "=== Step 2b: Joint Vis for all methods at 448x256 (including AdaIR) ==="
python3 tools/eval_all_methods_lowres.py \
    --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
    --eval_size 448 256 \
    --output_json experiment/eval/comparison_lowres_448x256.json

echo ""
echo "=== Step 3: Update all audit tables ==="
python3 tools/update_adair_tables.py \
    --adair_json experiment/eval/adair_eval.json \
    --lowres_json experiment/eval/comparison_lowres_448x256.json

echo ""
echo "=== Step 4: Final comparison table ==="
python3 -c "
import json
d = json.load(open('experiment/eval/comparison_lowres_448x256.json'))
print()
print(f'All methods at 448x256 (n=152 each)')
print(f'{\"Method\":<30} {\"PSNR\":>8} {\"SSIM\":>8} {\"MAE\":>8}')
print('-'*60)
for m, r in d['methods'].items():
    print(f'{m:<30} {r[\"psnr\"]:>8.4f} {r[\"ssim\"]:>8.4f} {r[\"mae\"]:>8.4f}')
"

echo ""
echo "=== Done! Run 'git add -A && git commit' to save results ==="
