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
