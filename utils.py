import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Tuple[Any, Any]]):
    """
    Detection models expect a list of images and a list of targets.
    """
    return tuple(zip(*batch))


def save_checkpoint(state: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0)
    return model, optimizer, start_epoch


class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

