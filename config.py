from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Config:
    """
    Top-level configuration for training and inference.

    Adjust these values directly or by passing CLI args in train.py / infer.py.
    """

    # Data
    train_data_dir: Path = Path("data/train")
    val_data_dir: Path = Path("data/val")
    classes: List[str] = field(default_factory=lambda: ["__background__", "object"])
    image_size: Tuple[int, int] = (800, 800)
    pretrained_backbone: bool = False

    # Optimization
    batch_size: int = 2
    num_epochs: int = 10
    learning_rate: float = 0.0025
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_steps: Tuple[int, ...] = (8,)
    lr_gamma: float = 0.1
    grad_clip_norm: Optional[float] = None

    # Runtime
    device: str = "cuda"  # set to "cpu" to force CPU training
    num_workers: int = 2
    seed: int = 1337
    print_freq: int = 20
    output_dir: Path = Path("outputs")
    checkpoint_path: Optional[Path] = None  # path to start from
    amp: bool = True

    # Inference
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    max_detections: int = 100

    def num_classes(self) -> int:
        # Mask R-CNN expects background class at index 0
        return len(self.classes)
