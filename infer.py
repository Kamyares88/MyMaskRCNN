import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

from config import Config
from model import load_for_inference


def _draw_predictions(image: Image.Image, boxes, masks, labels, scores, cfg: Config) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for box, mask, label, score in zip(boxes, masks, labels, scores):
        color = (255, 0, 0, 80)
        draw.rectangle(list(box), outline="red", width=3)
        text = f"{cfg.classes[label]} {score:.2f}" if label < len(cfg.classes) else f"{label} {score:.2f}"
        draw.text((box[0] + 3, box[1] + 3), text, fill="white")

        # apply mask overlay
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(image.size)
        overlay.paste(Image.new("RGBA", image.size, color), mask=mask_img)
    return overlay


def run_inference(
    model: torch.nn.Module, image_paths: List[Path], output_dir: Path, device: str, cfg: Config
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    results = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        tensor = F.to_tensor(image).to(device)
        with torch.no_grad():
            output = model([tensor])[0]

        keep = output["scores"] >= cfg.score_threshold
        boxes = output["boxes"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        masks = (output["masks"][keep] > cfg.mask_threshold).squeeze(1).cpu().numpy()

        overlay = _draw_predictions(image, boxes, masks, labels, scores, cfg)
        overlay.save(output_dir / f"{img_path.stem}_pred.png")

        result = {
            "image": str(img_path),
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }
        results.append(result)
        with open(output_dir / f"{img_path.stem}_pred.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved predictions for {img_path.name}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Mask R-CNN model")
    parser.add_argument("--images", type=Path, required=True, help="Path to image file or directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.score_threshold is not None:
        cfg.score_threshold = args.score_threshold
    if args.mask_threshold is not None:
        cfg.mask_threshold = args.mask_threshold
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_for_inference(cfg, checkpoint_path=str(args.checkpoint), device=device)

    image_paths = []
    if args.images.is_dir():
        image_paths = [
            p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    else:
        image_paths = [args.images]

    run_inference(model, image_paths, args.output_dir, device, cfg)


if __name__ == "__main__":
    main()

