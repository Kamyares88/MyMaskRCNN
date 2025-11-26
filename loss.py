from typing import Dict, Tuple

import torch


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Sum individual losses into a scalar while also returning detached scalars for logging.
    """
    loss_value = sum(loss for loss in loss_dict.values())
    loss_log = {k: v.item() for k, v in loss_dict.items()}
    loss_log["loss_total"] = loss_value.item()
    return loss_value, loss_log

