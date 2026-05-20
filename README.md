# IAPL AIGC Detection

IAPL is an image-level AIGC detector.  
This repo provides a Gradio demo (`demo.py`).

---

## Quick Start (3 Steps)

### Step 1) 环境安装

> 下面是安装方式（推荐复现一致环境）。

```bash
conda create -n iapl python=3.11 -y
conda activate iapl

python -m pip install --upgrade pip==26.1.1 setuptools==81.0.0 wheel==0.47.0
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

> 如果你只想快速运行，也可以尝试：

```bash
pip install -r requirements.txt
```

---

### Step 2) 下载权重（CLIP + IAPL 模型权重）

`demo.py` 默认从 `weights/` 目录读取权重，因此请准备如下文件：

```text
weights/ViT-L-14.pt
weights/checkpoint_best_acc_sd14.pth
```

推荐用 ModelScope 下载：

```bash
pip install modelscope
mkdir -p weights pretrained

python - <<'PY'
from modelscope import snapshot_download
snapshot_download("yihengli/IAPL_pretrain", local_dir="./pretrained")
PY

# CLIP 权重
cp ./pretrained/ViT-L-14.pt ./weights/ViT-L-14.pt

# IAPL 模型权重
cp ./pretrained/checkpoint_best_acc_sd14.pth ./weights/checkpoint_best_acc_sd14.pth
```


---

### Step 3) 运行

#### 3.1 启动 Gradio Demo

```bash
python demo.py
```

启动后终端会打印本地地址（如 `http://localhost:7860/`）以及可用外网地址（若可创建隧道）。

#### 3.2 （可选）命令行推理

```bash
python inference.py \
  --checkpoint weights/checkpoint_best_acc_sd14.pth \
  --clip_path weights/ViT-L-14.pt \
  --image_dir test_images \
  --output results.json
```

---

# AIGC 检测模型在笔记/文档场景下的表现调研总结

## 调研背景

近期 AI 生成内容（AIGC）在笔记、文档等办公场景中的应用日益广泛，许多用户使用 GPT、Gemini、Midjourney、Seedream 等工具生成笔记手写图、扫描文档等。这些场景下的 AIGC 图像具有高纹理密度、文字区域丰富、视觉上与真实图像高度相似等特点，对现有 AIGC 检测模型提出了新的挑战。

本次调研选取了 2025-2026 年间发表在顶级会议/期刊上的四个代表性 AIGC 检测方法，在自制测试集上进行评估。测试集包含 **19 张图片**：**6 张真实拍摄/扫描的文档笔记图片**（文件名以 `true` 开头）和 **13 张 AI 生成的文档笔记图片**（涵盖 GPT-5.4、Gemini 3 Pro、Seedream 等生成器，以 `doc_` 或 `note_` 开头）。

---

## 四个模型概述

| 模型 | 来源 | 核心方法 |
|------|------|----------|
| **WaRPAD** | NeurIPS 2025 | Training-free，基于 DINOv2 特征的 crop robustness + wavelet 高频扰动，比较原图与扰动后的 patch 相似度 |
| **PoundNet** | TPAMI 2026 | CLIP-ViT-L 主干，非对称 prompt learning 进行 real/fake 二分类，辅以类别感知监督 |
| **IAPL** | CVPR 2026 | CLIP + 频域先验增强（DCT 条件模块 + SRM 富隐写分析滤波器），利用频域信息提升泛化能力 |
| **LOTA** | ICCV 2025 | ResNet-50，从图像最低 3 位平面提取噪声特征，结合最大梯度 patch 选择策略进行二分类 |

### 1. WaRPAD（NeurIPS 2025）

**《Training-free Detection of AI-Generated Images via Cropping Robustness》**

- **无需训练**，直接使用预训练的 DINOv2 ViT-L/14 模型
- 核心假设：真实图像在 crop 操作下保持良好的 patch 一致性，而 AI 生成图像在不同 crop 之间的语义一致性较差
- 流程：对图像做 wavelet（Haar）高频扰动 → 随机 crop 16 个 patch → 计算 patch 间的 DINO 特征相似度 → 以平均相似度与阈值比较
- 默认阈值：0.93（相似度 > 0.93 认为是真实图像）

### 2. PoundNet（TPAMI 2026）

**《Penny-Wise and Pound-Foolish in AI-Generated Image Detection》**

- 基于 **CLIP ViT-L** 的监督学习方法
- 提出**非对称 prompt learning**：为 real/fake 两类设计不同的 prompt 上下文长度，利用 CLIP 的图文对齐能力进行分类
- 引入**类别感知辅助监督**，在训练时同时利用生成器类别信息提升特征判别力
- 在 ProGAN、DIF、DiffusionForensics 等多源数据集上训练

### 3. IAPL（CVPR 2026）

**CLIP + 频域先验增强**

- 基于 **CLIP** 模型，引入**频域先验（Frequency Prior）**
- 关键创新：在 CLIP 视觉编码器前端加入 **SRM（Spatial Rich Model）富隐写分析滤波器**和 **DCT 条件模块**，增强模型对生成痕迹的感知能力
- 使用 wavelet（DWT）分解，多尺度捕捉频域伪造痕迹
- 支持 ensemble：多个 checkpoint 融合推理，采用"任一模型判为 AIGC 即判为 AIGC"的策略

### 4. LOTA（ICCV 2025）

**《LOTA: Bit-Planes Guided AI-Generated Image Detection》**

- 基于 **ResNet-50**（ImageNet 预训练），二分类头
- 核心创新：从 RGB 各通道提取**最低 3 个比特位平面（bit-planes）**，scale 到 0-255
- 设计**最大梯度 patch 选择**策略（heuristic），选择纹理变化最剧烈的 patch
- 在 GenImage 的 SDv1.5 子集上训练，论文报告 98.9% 的跨生成器泛化准确率

---

## 实验结果对比

### 汇总表

| 模型 | 来源 | 总准确率 | Real 准确率 | Fake 准确率 | 备注 |
|------|------|----------|-------------|-------------|------|
| **WaRPAD** | NeurIPS 2025 | 31.6% (6/19) | 100% (6/6) | 0% (0/13) | 全部判为真实图像 |
| **PoundNet** | TPAMI 2026 | 31.6% (6/19) | 100% (6/6) | 0% (0/13) | 使用默认阈值 0.5 |
| **PoundNet†** | TPAMI 2026 | 63.2% (12/19) | 16.7% (1/6) | 84.6% (11/13) | 使用最佳搜索阈值 3.96e-06 |
| **LOTA** | ICCV 2025 | 31.6% (6/19) | 100% (6/6) | 0% (0/13) | 全部判为真实图像 |
| **IAPL** (ensemble) | CVPR 2026 | **52.6% (10/19)** | 83.3% (5/6) | 38.5% (5/13) | 唯一具备一定甄别能力的模型 |

> PoundNet† 使用了在测试集上搜索得到的"最优"阈值，阈值为 3.96e-06（而非默认 0.5），将 13 张 fake 中的 11 张正确识别，但代价是 6 张 real 中仅 1 张被正确保留。这是一种极端的 trade-off，实际部署中难以接受。

### 详细分析

#### WaRPAD — 完全失效

- 所有 19 张图片的 similarity score 都在 0.991-0.999 之间，远高于 0.93 阈值
- 原因：笔记/文档类图片整体纹理均匀、填充密集，crop 后各 patch 之间天然具有高相似度，AI 生成的文档图片在这方面与真实图片**没有显著差异**
- 该方法的"crop robustness"假设在文档场景下不成立

#### PoundNet — 默认阈值完全失效

- 使用默认阈值 0.5 时，所有图片的 fake_prob 均远低于 0.5（最高仅 0.0175），全部判为真实
- 即使使用极端低阈值（~4e-06），虽然能召回大部分 fake，但大幅牺牲 real 准确率
- fake_prob 分布极度压缩在 0~0.02 区间，real 和 fake 的分布严重重叠（mean_fake_prob: 0.0016 vs mean_real_prob_按反向推算 ~0.0001）
- CLIP-based 模型在自然图像数据集上训练的判别边界对文档场景完全偏移

#### LOTA — 完全失效

- 所有 19 张图片的 prob_real 均 > 0.87（绝大多数 > 0.99），全部判为真实
- 原因：笔记/文档类图片的最低 3 位平面与自然图像差异巨大。文档图片通常有大面积白色/浅色背景和黑色文字，其低位噪声模式与自然图像的随机噪声完全不同
- 该模型在 GenImage（自然图像数据集）上训练，其"位平面噪声 = 生成痕迹"的假设在文档场景下不适用

#### IAPL — 勉强可用

- **唯一一个在文档场景下具备一定甄别能力的模型**
- 总准确率 52.6%，虽不算高，但远超其他三个模型（均接近随机或完全偏向一侧）
- **Real 准确率 83.3%**（6 张真实图片中 5 张正确），表现尚可
- **Fake 准确率 38.5%**（13 张 AI 图片中 5 张被正确识别），仍有较大提升空间
- IAPL 表现优于其他模型的关键因素可能是：
  - **SRM 富隐写分析滤波器**能捕捉到 AI 生成过程中的微观像素统计异常
  - **DCT 频域条件模块**对 JPEG/文档类图像的频域特征更敏感
  - Ensemble 机制（SDv1.4 + ProGAN 双 checkpoint）提供了一定的互补性

### 按图片维度的 IAPL 详细表现

| 图片 | 真实标签 | IAPL 判断 | IAPL AIGC 概率 | 是否正确 |
|------|----------|-----------|-----------------|----------|
| true1_doc.jpg | Real | Real | 0.063 | ✓ |
| true2_doc.jpg | Real | Real | 0.059 | ✓ |
| true3_doc.jpg | Real | Real | 0.078 | ✓ |
| true4_note.jpg | Real | Real | 0.292 | ✓ |
| true5_note.jpg | Real | Real | 0.070 | ✓ |
| true6_note.jpg | Real | **AIGC** | 0.531 | ✗ |
| doc_banana.jpeg | Fake | **AIGC** | 0.761 | ✓ |
| doc_banana_3.jpeg | Fake | Real | 0.066 | ✗ |
| doc_gemini3.jpeg | Fake | Real | 0.121 | ✗ |
| doc_gpt54.jpeg | Fake | **AIGC** | 0.716 | ✓ |
| doc_seedream.jpeg | Fake | Real | 0.272 | ✗ |
| note_banana.jpeg | Fake | Real | 0.072 | ✗ |
| note_gemini3pro.jpeg | Fake | **AIGC** | 0.769 | ✓ |
| note_gpt54.jpeg | Fake | **AIGC** | 0.914 | ✓ |
| note_gpt54_2.jpeg | Fake | **AIGC** | 0.776 | ✓ |

> IAPL 对 GPT-5.4 生成的文档图像（doc_gpt54, note_gpt54, note_gpt54_2）和部分 Banana 生成器图片（doc_banana）检测能力较强，但对 Seedream、Midjourney 等生成器的检测能力较弱。

---

## 整体表现差的根本原因分析

### 1. 训练数据分布偏移（Distribution Shift）

所有四个模型均在**自然图像数据集**（GenImage、ProGAN、DIF 等）上训练/校准，这些数据集包含的是动物、风景、物体等自然场景图像。笔记/文档类图像具有完全不同的统计特性：

- **高频文字纹理**：大量边缘、笔画、线条
- **大面积白色/浅色背景**：像素值分布极度偏态
- **结构性排版**：密集但规则的文字排列
- **色彩分布单一**：通常只有黑、白、少量彩色

### 2. 方法假设在文档场景下失效

| 方法 | 核心假设 | 为什么在文档场景失效 |
|------|----------|----------------------|
| WaRPAD | crop 后 patch 相似度反映生成痕迹 | 文档图片的 patch 天然高相似（全是大面积背景+类似文字），不具备区分度 |
| PoundNet | CLIP 图文对齐+prompt 学习 | 训练数据无文档图像，CLIP 特征空间中文档图片的判别边界完全偏移 |
| LOTA | 低位平面噪声反映生成过程 | 文档图片的低位平面捕获的是文字笔画边缘的二值化特征，而非随机噪声 |
| IAPL | 频域+SRM 隐写分析特征 | 部分有效但不够充分，对多种生成器的泛化仍不足 |

### 3. 阈值/校准问题

- WaRPAD 的 0.93 阈值、PoundNet 的 0.5 阈值、LOTA 的 0.5 阈值都是在自然图像验证集上确定的，直接迁移到文档场景导致严重误判
- PoundNet 使用极端低阈值后虽能召回 fake，但 real 误判率激增，说明模型对文档图片的输出分布整体偏移，无法通过简单调阈值解决

### 4. 文档/笔记场景的特殊挑战

与自然照片不同，AI 生成的文档图片和真实扫描/拍摄的文档图片在以下维度上更难区分：
- 文字内容本身是生成的目标（而非附属物），生成模型对文字的渲染能力越来越强
- 纸张纹理、光照等"真实性线索"在低分辨率/web 图片中不容易体现
- 文档图片通常经过 JPEG 压缩（进一步抹去生成痕迹），而训练数据多为未压缩/轻度压缩图像
