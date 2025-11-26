from typing import Callable, Dict, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image: Image.Image, target: Dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict):
        image = F.to_tensor(image)
        if "masks" in target and isinstance(target["masks"], Image.Image):
            target["masks"] = torch.as_tensor(F.pil_to_tensor(target["masks"]) > 0, dtype=torch.uint8)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob: float = 0.5):
        self.prob = flip_prob

    def __call__(self, image: torch.Tensor, target: Dict):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
            width = image.shape[-1]
            boxes = target["boxes"]
            # flip boxes
            xmin = width - boxes[:, 2]
            xmax = width - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            target["boxes"] = boxes
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class Resize:
    """
    Resize image and masks keeping aspect ratio by default. Pads shorter side.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, target: Dict):
        orig_w, orig_h = image.size
        # PIL expects (width, height)
        image = image.resize(self.size, Image.BILINEAR)
        if "masks" in target:
            masks = target["masks"].float().unsqueeze(1)  # N x 1 x H x W
            target["masks"] = transforms.functional.resize(
                masks, size=self.size[::-1], interpolation=transforms.InterpolationMode.NEAREST
            ).squeeze(1).to(dtype=torch.uint8)

        # scale boxes
        if "boxes" in target:
            scale_w = self.size[0] / orig_w
            scale_h = self.size[1] / orig_h
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
            target["boxes"] = boxes
        return image, target


def build_transforms(train: bool, image_size: Tuple[int, int], with_augs: bool = True) -> Callable:
    tfs = [Resize(image_size), ToTensor()]
    if train and with_augs:
        tfs.append(RandomHorizontalFlip(0.5))
    return Compose(tfs)
