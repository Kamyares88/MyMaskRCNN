import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import Config
from dataset import InstanceSegmentationDataset
from engine import evaluate, train_one_epoch
from model import build_model
from transforms import build_transforms
from utils import collate_fn, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN from scratch")
    parser.add_argument("--train-data", type=Path, default=None, help="Path to training data root")
    parser.add_argument("--val-data", type=Path, default=None, help="Path to validation data root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use a pretrained ResNet50-FPN backbone")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.train_data:
        cfg.train_data_dir = args.train_data
    if args.val_data:
        cfg.val_data_dir = args.val_data
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.device:
        cfg.device = args.device
    if args.no_amp:
        cfg.amp = False
    if args.resume:
        cfg.checkpoint_path = args.resume
    if args.pretrained_backbone:
        cfg.pretrained_backbone = True

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    train_ds = InstanceSegmentationDataset(cfg.train_data_dir, transforms=build_transforms(True, cfg.image_size), classes=cfg.classes)
    val_ds = InstanceSegmentationDataset(cfg.val_data_dir, transforms=build_transforms(False, cfg.image_size, with_augs=False), classes=cfg.classes)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn
    )

    model = build_model(cfg)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=list(cfg.lr_steps), gamma=cfg.lr_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    start_epoch = 0
    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from checkpoint {cfg.checkpoint_path} at epoch {start_epoch}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, cfg, scaler)
        val_loss = evaluate(model, val_loader, device, cfg)
        scheduler.step()

        ckpt_path = cfg.output_dir / f"model_epoch_{epoch+1}.pth"
        save_checkpoint(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "config": cfg},
            ckpt_path,
        )
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} saved={ckpt_path}")


if __name__ == "__main__":
    main()
