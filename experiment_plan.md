# 补实验计划 (Experiment Plan)

**生成日期**: 2026-06-02  
**最后更新**: 2026-06-03  
**状态**: 第一轮最小补实验计划（已完成EXP-101~203；AdaIR EXP-202进行中）

---

## 前提条件检查

在运行任何实验之前，必须确认以下资源是否可用：

### 必须项（不可缺少）

- [✅] **LUCIDMine 微调权重**：Modal A10G 训练完成，best.pth (epoch 48)，val_psnr=23.14 dB
- [🔴] **矿山配对测试集（288对）**：**不存在**（作者确认未保存）；代理集(n=152)为最终评估基准
- [✅] **manifest CSV文件**：data/mine_manifest.csv 已生成

### 可选但推荐

- [✅] **chellocarol/lucidmine-40-video-dataset**：已访问（GitHub），n=152 test pairs
- [ ] **RIDCP代码和权重**：用于对比实验
- [ ] **合成压力测试集（30对）**：用于消融实验
- [ ] **GPU环境（推荐NVIDIA RTX 3090）**：用于FPS/时延测量

---

## 优先级 1：论文主表格核心指标（表2）

**目标**：复现288对真实配对测试集的PSNR/SSIM/MAE/Vis指标

### 步骤1a：准备LUCIDMine测试推理

```bash
# 需要：LUCIDMine微调权重、测试集输入图像、参考图像

python3 tools/infer_folder_ckpt.py \
  --checkpoint_path /path/to/lucidmine_finetuned.pth \
  --model_arch lucidmine \
  --state_key model \
  --input_dir /path/to/mine_test/input \
  --output_dir ./experiment/eval_P1/lucidmine_outputs \
  --output_ext png
```

### 步骤1b：运行全参考评估

```bash
# 选项A：使用manifest CSV（推荐）
python3 tools/eval_mine_per_video.py \
  --checkpoint_path /path/to/lucidmine_finetuned.pth \
  --manifest_path /path/to/mine_manifest.csv \
  --model_arch lucidmine \
  --state_key model \
  --output_json ./experiment/eval_P1/lucidmine_eval.json \
  --output_csv ./experiment/eval_P1/lucidmine_eval.csv
```

### 步骤1c：运行对比基线

```bash
# 骨干基线（无VCA/GARC）
python3 tools/infer_folder_ckpt.py \
  --checkpoint_path weights/Student.pth \
  --model_arch student \
  --state_key model \  # 注意：需确认Student.pth的key格式
  --input_dir /path/to/mine_test/input \
  --output_dir ./experiment/eval_P1/student_outputs

# 或直接用eval脚本
python3 tools/eval_mine_per_video.py \
  --checkpoint_path weights/Student.pth \
  --manifest_path /path/to/mine_manifest.csv \
  --model_arch student \
  --output_json ./experiment/eval_P1/student_eval.json \
  --output_csv ./experiment/eval_P1/student_eval.csv
```

**预期输出**：`experiment/eval_P1/` 目录下的 JSON 和 CSV 文件

---

## 优先级 2：经典方法对比基线

**目标**：在同一测试集上运行DCP、CLAHE、Retinex、RIDCP

### 步骤2：创建经典方法评估脚本

需要创建 `tools/eval_classical_baselines.py`，实现：

```python
# CLAHE（OpenCV内置，最简单实现）
import cv2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 论文设置：clipLimit=2.0，分块大小8×8，逐通道

# Retinex（多尺度，高斯尺度15/80/250，颜色恢复因子125）
# 需实现 MSRCR

# DCP（暗通道先验，窗口15×15，透射率下限0.1）
# 需实现 DCP

# RIDCP（需要RIDCP仓库和权重）
```

```bash
python3 tools/eval_classical_baselines.py \
  --input_dir /path/to/mine_test/input \
  --target_dir /path/to/mine_test/target \
  --output_dir ./experiment/eval_P2/ \
  --methods clahe dcp retinex \
  --output_csv ./reproduction_results.csv
```

**预期输出**：包含所有方法PSNR/SSIM/MAE的CSV

---

## 优先级 3：参数量与FLOPs精确验证

**目标**：与论文表1核对参数量、FLOPs、FPS

### 步骤3a：参数量验证（当前可执行）

```bash
cd /home/user/LUCID && python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
import torch
from model import Student, LUCIDMine
from thop import profile

# 论文声称的分辨率（待确认）
for model_class, name in [(Student, 'Student'), (LUCIDMine, 'LUCIDMine')]:
    model = model_class()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 3, 256, 256)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    print(f'{name}: {params:.4f}M params, {macs/1e9:.3f}G MACs @256x256')
EOF
```

**已知结果（res_blocks=1，ckpt匹配）**：
- Student: 2.998M params（论文声称6.42M）
- LUCIDMine: 3.015M params（论文声称6.48M）
- FLOPs @1080p: 88.5G（论文声称48.7G）

### 步骤3b：FPS测量（需GPU）

```bash
python3 - <<'EOF'
import torch, time, sys
sys.path.insert(0, '.')
from model import LUCIDMine

model = LUCIDMine().cuda().eval()
# Load checkpoint
state_dict = torch.load('weights/Student.pth', map_location='cuda')
model.load_state_dict(state_dict, strict=False)

x = torch.randn(1, 3, 1080, 1920).cuda()
# 预热
for _ in range(10): model(x)
torch.cuda.synchronize()

# 测量100次
start = time.time()
N = 100
for _ in range(N): model(x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f'FPS: {N/elapsed:.1f}, Latency: {elapsed/N*1000:.1f}ms')
EOF
```

**注意**：本环境无RTX 3090，FPS结果可能与论文不同。

---

## 优先级 4：Vis 综合可见度指标实现

**目标**：实现论文式(16)的Vis指标，验证表2/3/4/5/6中的Vis数值

```python
# 需实现的Vis指标（论文§3.3）
def compute_vis(image):
    """
    Vis = 0.25*C~ + 0.25*E~ + 0.20*(1-D~) + 0.15*(1-G~) + 0.15*S~
    C~: 归一化对比度
    E~: 归一化Shannon熵
    D~: 归一化暗度
    G~: 归一化眩光
    S~: 归一化梯度锐度
    """
    pass  # 需要实现归一化方案
```

创建脚本 `tools/eval_vis_metric.py` 并与参考实现对比。

**实现命令**：

```bash
python3 tools/eval_vis_metric.py \
  --input_dir ./experiment/eval_P1/lucidmine_outputs \
  --output_csv ./experiment/eval_P4/vis_results.csv
```

---

## 优先级 5：消融实验验证

**目标**：验证表5中VCA和GARC各自的贡献

**前提**：需要分别训练 VCA-only 和 GARC-only 变体

```bash
# 首先修改 LUCIDMine 以支持消融配置
# 方案：添加 use_vca, use_garc 参数

# 骨干基线（无VCA/GARC）= Student模型
python3 tools/eval_mine_per_video.py \
  --checkpoint_path weights/Student.pth \
  --model_arch student \
  --manifest_path /path/to/mine_manifest.csv \
  --output_json ./experiment/ablation/backbone_eval.json

# VCA-only（需要VCA-only微调权重）
# GARC-only（需要GARC-only微调权重）
# 完整LUCIDMine（需要完整微调权重）
```

---

## 优先级 6：40视频数据集评估

**目标**：使用 chellocarol/lucidmine-40-video-dataset 进行视频级评估

```bash
# 步骤1：检查数据集访问
# 使用 GitHub MCP 工具检查 chellocarol/lucidmine-40-video-dataset

# 步骤2：下载数据集
# 按照数据集说明下载

# 步骤3：构建manifest
python3 tools/build_mine_manifest.py \
  --data_dir /path/to/lucidmine-40-video-dataset \
  --output_csv ./data/mine_manifest_40video.csv

# 步骤4：运行LUCIDMine推理和评估
python3 tools/eval_mine_per_video.py \
  --checkpoint_path /path/to/lucidmine_finetuned.pth \
  --manifest_path ./data/mine_manifest_40video.csv \
  --model_arch lucidmine \
  --output_json ./experiment/video_eval/lucidmine_40video.json \
  --output_csv ./experiment/video_eval/lucidmine_40video.csv
```

---

## 当前可立即执行的实验

以下实验**无需额外数据集或权重**，可以立即执行：

### 立即可执行1：参数量和FLOPs精确验证

```bash
cd /home/user/LUCID && python3 tools/verify_model_complexity.py
```
（需先创建此脚本）

### 立即可执行2：Student.pth 零样本推理测试

```bash
# 下载或使用任意一张矿山图像测试推理流程
python3 tools/infer_folder_ckpt.py \
  --checkpoint_path weights/Student.pth \
  --model_arch lucidmine \
  --input_dir /tmp/test_images \
  --output_dir /tmp/lucidmine_output \
  --state_key model
```

### 立即可执行3：Vis指标实现与自测

```bash
# 实现Vis指标并用测试图像验证
python3 tools/eval_vis_metric.py --help
```

---

## 实验依赖汇总

| 实验 | 所需权重 | 所需数据 | 所需代码 | 优先级 |
|-----|---------|---------|---------|------|
| 参数量/FLOPs验证 | Student.pth ✅ | 无 | 已有 ✅ | P3 |
| Student零样本推理 | Student.pth ✅ | 任意图像 | 已有 ✅ | P2 |
| LUCIDMine测试集评估 | LUCIDMine微调权重 ❌ | 测试集 ❌ | 已有 ✅ | P1 |
| CLAHE基线 | 无 | 测试集 ❌ | 需开发 | P2 |
| DCP/Retinex基线 | 无 | 测试集 ❌ | 需开发 | P2 |
| RIDCP基线 | RIDCP权重 ❌ | 测试集 ❌ | 需获取 | P2 |
| 消融实验 | 变体权重 ❌ | 30对合成集 ❌ | 需开发 | P3 |
| Vis指标验证 | 任意权重 | 输出图像 | 需开发 | P4 |
| 40视频评估 | LUCIDMine权重 ❌ | 40视频集 ⚠️ | 已有 ✅ | P6 |

**最小可执行实验**：参数量/FLOPs验证（已在本报告中完成）

**关键缺口**：LUCIDMine微调权重 + 矿山配对测试集，缺少这两项则无法验证论文主要贡献。
