import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import InstanceSegmentationDataset
from model import load_for_inference
from transforms import build_transforms
from utils import collate_fn, set_seed


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU for two boxes defined as [x1, y1, x2, y2].
    """
    xa1, ya1, xa2, ya2 = [float(v) for v in box1]
    xb1, yb1, xb2, yb2 = [float(v) for v in box2]
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def evaluate_image(pred: Dict, target: Dict, iou_thresh: float = 0.5):
    # sort predictions by score descending
    order = torch.argsort(pred["scores"], descending=True)
    pred_boxes = pred["boxes"][order]
    pred_labels = pred["labels"][order]
    pred_scores = pred["scores"][order]

    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    matched_gt = set()

    tp = 0
    fp = 0
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        best_iou = 0.0
        best_idx = -1
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in matched_gt or label != gt_label:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def run_eval(cfg: Config, checkpoint: Path, data_dir: Path, device: str):
    set_seed(cfg.seed)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dataset = InstanceSegmentationDataset(data_dir, transforms=build_transforms(False, cfg.image_size, with_augs=False), classes=cfg.classes)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = load_for_inference(cfg, checkpoint_path=str(checkpoint), device=device)

    precisions = []
    recalls = []
    f1s = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                precision, recall, f1 = evaluate_image(output, target)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
    print(
        f"Eval over {len(dataset)} images | "
        f"precision@0.5={sum(precisions)/len(precisions):.3f} "
        f"recall@0.5={sum(recalls)/len(recalls):.3f} "
        f"f1@0.5={sum(f1s)/len(f1s):.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Mask R-CNN checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Path to validation dataset root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    run_eval(cfg, args.checkpoint, args.data, args.device)
