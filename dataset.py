import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def polygons_to_mask(polygons: Sequence[Sequence[Sequence[float]]], size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert polygons (list of points) into a binary mask tensor.
    """
    mask = Image.new("L", size, 0)
    for polygon in polygons:
        ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], outline=1, fill=1)
    return torch.as_tensor(np.array(mask), dtype=torch.uint8)


def _load_annotation(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _coerce_labels(raw_labels: Sequence, classes: Optional[List[str]]) -> torch.Tensor:
    if not raw_labels:
        return torch.zeros((0,), dtype=torch.int64)
    if isinstance(raw_labels[0], str):
        if not classes:
            raise ValueError("Annotation has string labels but no class list was provided.")
        name_to_idx = {name: idx for idx, name in enumerate(classes)}
        return torch.as_tensor([name_to_idx[label] for label in raw_labels], dtype=torch.int64)
    return torch.as_tensor(raw_labels, dtype=torch.int64)


class InstanceSegmentationDataset(Dataset):
    """
    Minimal dataset for Mask R-CNN.

    Expected folder layout:
    root/
      images/
        img_0001.jpg
      annotations/
        img_0001.json

    Each annotation JSON should include:
    {
      "boxes": [[x1,y1,x2,y2], ...],
      "labels": [1, 1, ...] or ["class_name", ...],
      "polygons": [ [ [x,y], [x,y], ... ], ... ]  # optional
      "iscrowd": [0, 0, ...]                       # optional
    }
    Masks are rasterized from polygons; provide multiple polygons per instance for disjoint shapes.
    """

    def __init__(
        self,
        root: Path,
        transforms=None,
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.transforms = transforms
        self.classes = classes

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Could not find images directory at {self.img_dir}")

        self.images = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        ann_path = self.ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file for {img_path.name}: {ann_path}")

        ann = _load_annotation(ann_path)
        boxes = torch.as_tensor(ann.get("boxes", []), dtype=torch.float32)
        labels = _coerce_labels(ann.get("labels", []), self.classes)
        polygons = ann.get("polygons", [])
        masks: List[torch.Tensor] = []
        for poly in polygons:
            if not poly:
                continue
            is_multipart = isinstance(poly[0], (list, tuple)) and isinstance(poly[0][0], (int, float))
            if is_multipart:
                mask = polygons_to_mask([poly], image.size)
            elif isinstance(poly[0], (list, tuple)):
                mask = polygons_to_mask(poly, image.size)
            else:
                raise ValueError("Polygon format not recognized; expected list of points or list of lists.")
            masks.append(mask)
        masks_tensor = torch.stack(masks, dim=0) if masks else torch.zeros((0, image.height, image.width), dtype=torch.uint8)
        iscrowd = torch.as_tensor(ann.get("iscrowd", [0] * len(labels)), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor([], dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
