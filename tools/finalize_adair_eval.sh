#!/bin/bash
# Finalize AdaIR evaluation after all 152 images are inferred.
# Run this after experiment/infer_test/adair_dehaze/ has 152 images.

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
echo "=== Step 1: Full lowres comparison (all methods at 448x256, incl. Vis) ==="
python3 tools/eval_all_methods_lowres.py \
    --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
    --eval_size 448 256 \
    --output_json experiment/eval/comparison_lowres_448x256.json

echo ""
echo "=== Step 2: AdaIR standalone PSNR/SSIM/MAE with std ==="
python3 tools/eval_adair_metrics.py \
    --pred_dir experiment/infer_test/adair_dehaze \
    --target_dir /home/user/lucidmine-40-video-dataset/data/test/target \
    --eval_size 448 256 \
    --output_json experiment/eval/adair_eval.json

echo ""
echo "=== Done! Key results ==="
python3 -c "
import json
print()
print('--- comparison_lowres_448x256.json ---')
d = json.load(open('experiment/eval/comparison_lowres_448x256.json'))
print(f'{\"Method\":<30} {\"PSNR\":>8} {\"SSIM\":>8} {\"MAE\":>8} {\"Vis\":>8}')
print('-'*66)
for m, r in d['methods'].items():
    print(f'{m:<30} {r[\"psnr\"]:>8.4f} {r[\"ssim\"]:>8.4f} {r[\"mae\"]:>8.4f} {r.get(\"vis\", float(\"nan\")):>8.4f}')
"
