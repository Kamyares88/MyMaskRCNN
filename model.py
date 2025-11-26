from typing import Optional

import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from config import Config


def build_model(cfg: Config) -> MaskRCNN:
    """
    Construct a Mask R-CNN model with a ResNet50-FPN backbone.
    """
    weights = "DEFAULT" if cfg.pretrained_backbone else None
    backbone = resnet_fpn_backbone("resnet50", weights=weights)
    model = MaskRCNN(backbone, num_classes=cfg.num_classes())

    # Replace predictors to match number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes())

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, cfg.num_classes())

    return model


def load_for_inference(cfg: Config, checkpoint_path: Optional[str] = None, device: str = "cpu") -> MaskRCNN:
    model = build_model(cfg)
    model.to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
    model.eval()
    return model
