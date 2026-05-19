"""
IAPL AIGC Detection Demo — Gradio Web UI
Supports: upload / drag-and-drop image detection, gallery of correctly predicted samples.
"""

import os
import tempfile

# Fix httpx incompatibility with no_proxy containing IPv6 entries like ::1
for _key in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_key, "")
    if _val:
        _parts = [p.strip() for p in _val.split(",") if "::" not in p]
        os.environ[_key] = ",".join(_parts)

import sys
import json
import torch
import argparse
import inspect

# 方法1：全局添加
torch.serialization.add_safe_globals([argparse.Namespace])
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = "/home/vlr/IAPL/test_images"
RESULTS_JSON = os.path.join(BASE_DIR, "results.json")
CLIP_CACHE = os.path.join(BASE_DIR, "weights/ViT-L-14.pt")

# ─── IAPL Model Loading ──────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from models import build_model


def _get_args_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cuda:0")
    args = ckpt["args"]
    defaults = {
        "smooth": False, "ema": False, "use_contrast": False,
        "phase_2": False, "ois": False, "loss_adapter": 1.0,
        "loss_contrast": 1.0, "loss_condition": 1.0,
        "selection_p": 0.2, "tta_steps": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.eval = True
    args.distributed = False
    args.tta = False
    args.resume = False
    args.pretrained_model = checkpoint_path
    if os.path.exists(CLIP_CACHE):
        args.clip_path = CLIP_CACHE
    return args, ckpt


def load_iapl_model(checkpoint_path, device):
    train_args, ckpt = _get_args_from_checkpoint(checkpoint_path)
    model = build_model(train_args)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((train_args.img_resolution, train_args.img_resolution)),
        transforms.CenterCrop(train_args.crop_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


# ─── Detector ─────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"Device: {self.device}")

        print("Loading IAPL SD1.4 model...")
        self.iapl_sd14, self.iapl_sd14_tf = load_iapl_model(
            os.path.join(BASE_DIR, "weights/checkpoint_best_acc_sd14.pth"), self.device)

        print("IAPL SD1.4 model loaded.")

    @torch.no_grad()
    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        img_sd14 = self.iapl_sd14_tf(img).unsqueeze(0).to(self.device)

        logit_sd14 = self.iapl_sd14(img_sd14)
        prob_sd14 = torch.sigmoid(logit_sd14).item()

        is_aigc = prob_sd14 > 0.5

        return {
            "sd14_prob": round(prob_sd14, 4),
            "verdict": "AIGC" if is_aigc else "Real",
        }



# ─── Gallery ──────────────────────────────────────────────────────────────────
def build_gallery_data():
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    real_prefix = data.get("real_prefix", "true")
    gallery = {"doc": [], "note": []}

    for r in data["results"]:
        path = r["image_path"]
        basename = os.path.basename(path)
        is_aigc = r["is_aigc"]
        prob = r["aigc_probability"]

        gt_is_aigc = not basename.lower().startswith(real_prefix.lower())
        if is_aigc != gt_is_aigc:
            continue

        if basename.startswith("doc_") or (basename.startswith("true") and "doc" in basename):
            category = "doc"
        else:
            category = "note"

        if not gt_is_aigc:
            caption = "Real"
        else:
            name_no_ext = os.path.splitext(basename)[0]
            parts = name_no_ext.split("_")
            source_raw = parts[1] if len(parts) >= 2 else "unknown"
            source_map = {
                "banana": "Banana",
                "gemini3": "Gemini3Pro",
                "gemini3pro": "Gemini3Pro",
                "gpt54": "GPT-4o",
                "seedream": "SeeDream",
            }
            caption = source_map.get(source_raw.lower(), source_raw)

        gallery[category].append({
            "path": path,
            "caption": caption,
            "prob": prob,
            "gt_is_aigc": gt_is_aigc,
        })

    return gallery


# ─── Gradio UI ────────────────────────────────────────────────────────────────
import gradio as gr

UI_THEME = gr.themes.Soft()
UI_CSS = """
.result-box { min-height: 120px; }
"""
BLOCKS_INIT_PARAMS = inspect.signature(gr.Blocks.__init__).parameters
LAUNCH_PARAMS = inspect.signature(gr.Blocks.launch).parameters


def create_demo():
    detector = Detector()
    gallery_data = build_gallery_data()

    def detect(image):
        if image is None:
            return "Please upload an image.", ""
        result = detector.predict(image)

        verdict = result["verdict"]
        prob = result["sd14_prob"]
        if verdict == "AIGC":
            summary = (
                f"## Detection Result: **AIGC** (AI-Generated)\n"
                f"### AIGC Confidence: {prob:.1%}"
            )
        else:
            summary = (
                f"## Detection Result: **Real** (Authentic)\n"
                f"### Real Confidence: {1 - prob:.1%}"
            )

        detail = (
            f"### Model Details\n"
            f"| Model | AIGC Probability |\n|---|---|\n"
            f"| IAPL-SD1.4 | {result['sd14_prob']:.1%} |"
        )

        return summary, detail

    def _iter_candidate_paths(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for key in ("path", "image", "name", "url"):
                if key in obj:
                    yield from _iter_candidate_paths(obj[key])
        elif isinstance(obj, (list, tuple)):
            for part in obj:
                yield from _iter_candidate_paths(part)

    def select_example(evt: gr.SelectData, gallery_items):
        candidates = []
        selected = getattr(evt, "value", None)
        candidates.extend(list(_iter_candidate_paths(selected)))

        selected_index = getattr(evt, "index", None)
        if isinstance(selected_index, (list, tuple)) and len(selected_index) > 0:
            selected_index = selected_index[0]
        if isinstance(selected_index, int) and 0 <= selected_index < len(gallery_items):
            candidates.insert(0, gallery_items[selected_index][0])

        image_path = next(
            (
                p for p in candidates
                if isinstance(p, str) and os.path.exists(p)
            ),
            None
        )
        if not image_path:
            return None, "Please select a valid example image.", ""

        img = Image.open(image_path).convert("RGB")
        summary, detail = detect(img)
        return img, summary, detail

    # Gallery items
    doc_items = []
    for item in gallery_data["doc"]:
        conf = item["prob"] if item["gt_is_aigc"] else (1 - item["prob"])
        tag = f"{item['caption']} ({'AIGC' if item['gt_is_aigc'] else 'Real'}, {conf:.1%})"
        doc_items.append((item["path"], tag))

    note_items = []
    for item in gallery_data["note"]:
        conf = item["prob"] if item["gt_is_aigc"] else (1 - item["prob"])
        tag = f"{item['caption']} ({'AIGC' if item['gt_is_aigc'] else 'Real'}, {conf:.1%})"
        note_items.append((item["path"], tag))

    def select_doc_example(evt: gr.SelectData):
        return select_example(evt, doc_items)

    def select_note_example(evt: gr.SelectData):
        return select_example(evt, note_items)

    blocks_kwargs = {"title": "IAPL AIGC Detection Demo"}
    if "theme" in BLOCKS_INIT_PARAMS and "theme" not in LAUNCH_PARAMS:
        blocks_kwargs["theme"] = UI_THEME
    if "css" in BLOCKS_INIT_PARAMS and "css" not in LAUNCH_PARAMS:
        blocks_kwargs["css"] = UI_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# IAPL AIGC Image Detection Demo\n"
            "Upload or drag-and-drop an image to detect whether it is AI-generated content (AIGC)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload / Drag & Drop Image",
                )
                detect_btn = gr.Button("Detect", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_summary = gr.Markdown("", elem_classes=["result-box"])
                detail_result = gr.Markdown("")

        detect_btn.click(
            fn=detect,
            inputs=input_image,
            outputs=[result_summary, detail_result],
        )
        input_image.upload(
            fn=detect,
            inputs=input_image,
            outputs=[result_summary, detail_result],
        )

        gr.Markdown(
            "---\n"
            "## Gallery: IAPL Correct Predictions\n"
            "Images that IAPL correctly identified. "
            "Real images labeled **Real**, AIGC images labeled with their source model."
        )

        gr.Markdown("### Document-style")
        doc_gallery = gr.Gallery(
            value=doc_items,
            columns=len(doc_items) if len(doc_items) > 0 else 1,
            height=300,
            object_fit="cover",
            label="Document",
        )

        gr.Markdown("### Note-style")
        note_gallery = gr.Gallery(
            value=note_items,
            columns=len(note_items) if len(note_items) > 0 else 1,
            height=300,
            object_fit="cover",
            label="Note",
        )

        doc_gallery.select(
            fn=select_doc_example,
            inputs=None,
            outputs=[input_image, result_summary, detail_result],
        )

        note_gallery.select(
            fn=select_note_example,
            inputs=None,
            outputs=[input_image, result_summary, detail_result],
        )

    return demo


if __name__ == "__main__":
    import subprocess
    import time
    import re

    demo = create_demo()

    def open_localtunnel(port=7860, timeout_sec=30):
        print("\nGradio share URL unavailable, trying localtunnel fallback...")
        lt_proc = subprocess.Popen(
            ["npx", "--yes", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        deadline = time.time() + timeout_sec
        public_url = None
        while time.time() < deadline:
            line = lt_proc.stdout.readline()
            if not line:
                time.sleep(0.5)
                continue
            m = re.search(r"(https://[a-z0-9-]+\.loca\.lt)", line)
            if m:
                public_url = m.group(1)
                break
        if public_url:
            return lt_proc, public_url
        if lt_proc.poll() is None:
            lt_proc.terminate()
        return None, None

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": True,
    }
    if "prevent_thread_lock" in LAUNCH_PARAMS:
        launch_kwargs["prevent_thread_lock"] = True
    if "theme" in LAUNCH_PARAMS:
        launch_kwargs["theme"] = UI_THEME
    if "css" in LAUNCH_PARAMS:
        launch_kwargs["css"] = UI_CSS

    _, local_url, public_url = demo.launch(**launch_kwargs)
    print(f"Local URL: {local_url}")

    def emit_external_url(url: str):
        # 单独打印纯 URL，便于直接复制/脚本读取
        print("\n" + url + "\n", flush=True)

    lt_proc = None
    if public_url:
        emit_external_url(public_url)
    else:
        lt_proc, public_url = open_localtunnel(port=7860, timeout_sec=30)
        if public_url:
            emit_external_url(public_url)
        else:
            print("Failed to get public URL from both Gradio share and localtunnel.")

    if launch_kwargs.get("prevent_thread_lock", False):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down demo...")
        finally:
            if lt_proc and lt_proc.poll() is None:
                lt_proc.terminate()
            demo.close()
