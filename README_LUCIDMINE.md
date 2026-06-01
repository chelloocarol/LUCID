# LUCIDMine CoA Adaptation

LUCIDMine is implemented as a lightweight extension of the CoA student backbone. It keeps the original compression-and-adaptation path intact and adds two identity-initialized mine-scene modules:

- `VisibilityConditionedCoAAdapter` (VCCA): computes luminance, dark-channel, glare, and low-visibility priors from the CLIP-normalized input after converting it back to RGB image space. The four-prior tensor is projected into the 128-channel compressed CoA bottleneck and added through a zero-initialized scalar gate.
- `GlareAwareResidualCalibrator` (GARC): calibrates the final restoration residual using glare and low-visibility priors. The calibrator is zero-gated, so `LUCIDMine` returns the same output as the underlying CoA student before adaptation.

The design goal is transfer safety without dead adapters: public CoA student checkpoints can be loaded with `strict=False`; the new gates start at zero to preserve the pretrained output, while the projection layers use small nonzero initialization so gradients can activate the adapters during fine-tuning.

## Inference

```powershell
$py='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
cd D:\ARIS\LUCIDMine\code\CoA_lucidmine
& $py tools\infer_folder_ckpt.py `
  --checkpoint_path model\Student_model\Student.pth `
  --model_arch lucidmine `
  --input_dir D:\ARIS\LUCIDMine\data\raw\mine_test `
  --output_dir D:\ARIS\LUCIDMine\data\processed\neural_lucidmine
```

If pretrained weights are unavailable, the paper-generation scripts use deterministic classical and CoA-style fallback outputs so that figures and metrics remain reproducible.
