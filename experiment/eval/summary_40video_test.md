# 40视频测试集实验结果汇总

**数据集**: lucidmine-40-video-dataset test split  
**测试集规模**: n=152 对  
**图像分辨率**: 960×540  
**参考图像**: RIDCP 伪标签（非真实清晰帧）  
**日期**: 2026-06-02

---

## 全参考指标 (vs RIDCP 伪标签)

> ⚠️ 注意：参考图像为 RIDCP 生成的伪清晰帧，RIDCP 方法在此数据集上有天然优势（与自身输出比较）

| 方法 | PSNR↑/dB | SSIM↑ | MAE↓ | 状态 |
|-----|---------|------|-----|-----|
| 输入原图 | 18.908 | 0.7549 | 0.0920 | ✅ 已跑 |
| CLAHE | 18.071 | 0.7755 | 0.1049 | ✅ 已跑 |
| Retinex (MSR) | 10.716 | 0.2366 | 0.2244 | ✅ 已跑 |
| DCP | 16.335 | 0.6323 | 0.1336 | ✅ 已跑 |
| Student (backbone, 未微调) | 18.783 | 0.7406 | 0.0995 | ✅ 已跑 |
| LUCIDMine (未微调，zero-gate) | **18.783** | **0.7406** | **0.0995** | ✅ 已跑（= Student，符合预期） |
| **LUCIDMine (Modal A10G, ep48)** | **23.014** | **0.8438** | **0.0635** | ✅ EXP-201 完成 |

## Vis 无参考指标

| 方法 | Vis↑ |
|-----|-----|
| 输入原图 | 0.526 |
| CLAHE | 0.656 |
| Retinex | 0.397 |
| DCP | **0.775** |
| Student (未微调) | 0.692 |
| LUCIDMine (未微调，zero-gate) | **0.692** |
| **LUCIDMine (微调后)** | ⏳ |

---

## 与论文声明值对比

> 论文使用不同数据集（1920对，1920×1080，真实配对），数值不可直接比较

| 指标 | 论文(CLAHE,n=288) | 本数据集(CLAHE,n=152) | 差值 |
|-----|----------------|-------------------|-----|
| PSNR | 16.48 | 18.07 | +1.59 |
| SSIM | 0.805 | 0.776 | -0.029 |
| MAE | 0.137 | 0.105 | -0.032 |

| 指标 | 论文(DCP,n=288) | 本数据集(DCP,n=152) | 差值 |
|-----|--------------|------------------|-----|
| PSNR | 15.92 | 16.34 | +0.42 |
| SSIM | 0.821 | 0.632 | -0.189 |

---

## 关键验证结论

1. ✅ **Zero-gate 初始化正确**: LUCIDMine(init) 指标 = Student(backbone)，符合论文设计
2. ✅ **训练流程验证**: 3-batch smoke test 成功（L1+SSIM梯度正常流动）
3. ⏳ **待GPU训练**: 需 Modal A10G 运行 100 epoch 才能验证核心指标声明
4. ⚠️ **数据集差异**: 本数据集(RIDCP伪标签target)与论文数据集(真实配对)不同，指标绝对值不可直接比较

---

## 下一步

```bash
# 配置 Modal token 后运行:
modal run modal_train.py

# 训练完成后自动评估:
python tools/eval_mine_per_video.py \
  --checkpoint_path experiment/LUCIDMine/modal_run/best.pth \
  --manifest data/mine_manifest.csv \
  --model_arch lucidmine \
  --output_json experiment/eval/lucidmine_final_eval.json \
  --output_csv experiment/eval/lucidmine_final_eval.csv
```
