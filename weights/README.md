# Weights

This directory contains the checkpoint used to initialize the LUCIDMine experiments.

## `Student.pth`

- Source path on the experiment machine: `D:\ARIS\COA\CoA-main\model\Student_model\Student.pth`
- File size: `12,108,677` bytes
- Original timestamp on the experiment machine: `2024-10-22 10:05:36`
- Usage: initialization checkpoint for the compact restoration student backbone before LUCIDMine adaptation modules are applied.

This checkpoint is not a fully trained LUCIDMine model. It is the backbone initialization used in the reliability-map ablation and related LUCIDMine adaptation experiments.

Example loading path:

```bash
--init_ckpt weights/Student.pth
```
