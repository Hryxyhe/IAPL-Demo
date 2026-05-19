"""
Inference script for IAPL AIGC detection.
Loads a trained checkpoint, runs on images in test_images/, writes results to JSON.
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Make imports work from the script's directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model


def get_args_from_checkpoint(checkpoint_path):
    """Extract training args from checkpoint, overriding what's needed for inference."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    # Fill in missing fields that may not exist in older checkpoints
    defaults = {
        "smooth": False,
        "ema": False,
        "use_contrast": False,
        "phase_2": False,
        "ois": False,
        "loss_adapter": 1.0,
        "loss_contrast": 1.0,
        "loss_condition": 1.0,
        "selection_p": 0.2,
        "tta_steps": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    # Override fields that are irrelevant or problematic for inference
    args.eval = True
    args.distributed = False
    args.tta = False
    args.resume = False
    args.pretrained_model = checkpoint_path
    return args, ckpt


def get_inference_transform(img_resolution=256, crop_resolution=224):
    """Standard eval transform matching the training pipeline."""
    return transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_images(image_dir):
    """Recursively collect image paths from a directory."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in valid_ext:
                images.append(os.path.join(root, f))
    return images


def is_real_image(filepath, real_prefix="true"):
    """Determine ground-truth label from filename. Files starting with real_prefix are real (label=0)."""
    basename = os.path.basename(filepath)
    return basename.lower().startswith(real_prefix.lower())


def evaluate_results(results, real_prefix="true"):
    """Compute evaluation metrics from inference results.

    Label convention: real=0, fake(AIGC)=1.
    Files whose basename starts with real_prefix are treated as real.
    """
    y_true, y_pred_prob = [], []
    for r in results:
        if r["aigc_probability"] is None:
            continue
        label = 0 if is_real_image(r["image_path"], real_prefix) else 1
        y_true.append(label)
        y_pred_prob.append(r["aigc_probability"])

    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ap = average_precision_score(y_true, y_pred_prob) if len(set(y_true)) > 1 else 0

    real_probs = y_pred_prob[y_true == 0]
    fake_probs = y_pred_prob[y_true == 1]
    real_acc = (real_probs <= 0.5).mean() if len(real_probs) > 0 else 0
    fake_acc = (fake_probs > 0.5).mean() if len(fake_probs) > 0 else 0

    return {
        "total": len(y_true),
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "average_precision": round(ap, 4),
        "real_accuracy": round(real_acc, 4),
        "fake_accuracy": round(fake_acc, 4),
        "mean_aigc_prob_real": round(float(real_probs.mean()), 4) if len(real_probs) > 0 else None,
        "mean_aigc_prob_fake": round(float(fake_probs.mean()), 4) if len(fake_probs) > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="IAPL AIGC Detection Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best_acc_sd14.pth",
                        help="Path to the trained checkpoint")
    parser.add_argument("--image_dir", type=str, default="test_images",
                        help="Directory containing images to test")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_path", type=str, default=None,
                        help="Path to ViT-L-14.pt CLIP weights (overrides checkpoint args)")
    parser.add_argument("--real_prefix", type=str, default="true",
                        help="Filename prefix for real images (default: 'true')")
    args = parser.parse_args()

    # Load checkpoint and recover training config
    train_args, ckpt = get_args_from_checkpoint(args.checkpoint)
    if args.clip_path is not None:
        train_args.clip_path = args.clip_path

    device = torch.device(args.device)

    # Build model
    model = build_model(train_args)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = get_inference_transform(
        img_resolution=train_args.img_resolution,
        crop_resolution=train_args.crop_resolution,
    )

    # Collect images
    if not os.path.isdir(args.image_dir):
        print(f"Error: image directory '{args.image_dir}' not found.")
        sys.exit(1)

    image_paths = collect_images(args.image_dir)
    if len(image_paths) == 0:
        print(f"Error: no images found in '{args.image_dir}'.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in '{args.image_dir}'.")

    # Inference
    results = []
    with torch.no_grad():
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Warning: failed to load {p}: {e}")
                    results.append({
                        "image_path": p,
                        "aigc_probability": None,
                        "is_aigc": None,
                        "error": str(e),
                    })

            if not batch_tensors:
                continue

            batch_input = torch.stack(batch_tensors).to(device)
            logits = model(batch_input)
            probs = logits.sigmoid().flatten().tolist()

            for path, prob in zip(valid_paths, probs):
                results.append({
                    "image_path": path,
                    "aigc_probability": round(prob, 6),
                    "is_aigc": prob > 0.5,
                })

            print(f"  Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")

    # Evaluate
    valid_results = [r for r in results if r["aigc_probability"] is not None]
    metrics = evaluate_results(results, real_prefix=args.real_prefix)

    if metrics:
        print(f"\n{'='*50}")
        print(f"Evaluation Results (real_prefix='{args.real_prefix}')")
        print(f"{'='*50}")
        print(f"  Total: {metrics['total']}  (Real: {metrics['num_real']}, Fake: {metrics['num_fake']})")
        print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1 Score:           {metrics['f1_score']:.4f}")
        print(f"  Average Precision:  {metrics['average_precision']:.4f}")
        print(f"  Real Accuracy:      {metrics['real_accuracy']:.4f}  (real images correctly classified)")
        print(f"  Fake Accuracy:      {metrics['fake_accuracy']:.4f}  (fake images correctly classified)")
        if metrics["mean_aigc_prob_real"] is not None:
            print(f"  Mean AIGC prob (real): {metrics['mean_aigc_prob_real']:.4f}")
        if metrics["mean_aigc_prob_fake"] is not None:
            print(f"  Mean AIGC prob (fake): {metrics['mean_aigc_prob_fake']:.4f}")
        print(f"{'='*50}")

    # Write results
    output = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "image_dir": os.path.abspath(args.image_dir),
        "real_prefix": args.real_prefix,
        "total_images": len(image_paths),
        "metrics": metrics,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
