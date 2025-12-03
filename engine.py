from typing import Dict, Iterable, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config
from loss import reduce_loss_dict
from utils import SmoothedValue


def _move_to_device(batch: Iterable, device: str):
    images, targets = batch
    images = [img.to(device) for img in images]
    converted = []
    for t in targets:
        converted.append({k: v.to(device) for k, v in t.items()})
    return images, converted


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    cfg: Config,
    scaler: GradScaler,
):
    model.train()
    losses_meter = SmoothedValue()
    for step, batch in enumerate(data_loader):
        images, targets = _move_to_device(batch, device)
        try:
            with autocast(enabled=cfg.amp):
                loss_dict = model(images, targets)
                loss, log_dict = reduce_loss_dict(loss_dict)
        except Exception as e:
            # Helpful debug info when a batch triggers a CUDA error
            print(f"[train] failure on batch {step}")
            print(f" num images: {len(images)}")
            for i, t in enumerate(targets):
                print(
                    f"  sample {i}: labels={t['labels'].tolist()} "
                    f"boxes_shape={tuple(t['boxes'].shape)} masks_shape={tuple(t['masks'].shape)} "
                    f"boxes_sample={t['boxes'][:2].tolist() if t['boxes'].numel() else []}"
                )
            raise

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        losses_meter.update(loss.item(), n=len(images))

        if step % cfg.print_freq == 0:
            log_str = f"Epoch {epoch} Step {step}/{len(data_loader)} | loss: {loss.item():.4f} "
            log_str += " ".join(f"{k}:{v:.3f}" for k, v in log_dict.items())
            print(log_str)
    return losses_meter.avg


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: str, cfg: Config):
    """
    Compute validation loss. Detection models only return losses when in training mode,
    so we temporarily switch to train() while keeping gradients off.
    """
    was_training = model.training
    model.train()
    losses_meter = SmoothedValue()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            images, targets = _move_to_device(batch, device)
            try:
                loss_dict = model(images, targets)
                loss, _ = reduce_loss_dict(loss_dict)
            except Exception:
                print(f"[eval] failure on batch {step}")
                for i, t in enumerate(targets):
                    print(
                        f"  sample {i}: labels={t['labels'].tolist()} "
                        f"boxes_shape={tuple(t['boxes'].shape)} masks_shape={tuple(t['masks'].shape)}"
                    )
                raise
            losses_meter.update(loss.item(), n=len(images))
    if not was_training:
        model.eval()
    return losses_meter.avg
