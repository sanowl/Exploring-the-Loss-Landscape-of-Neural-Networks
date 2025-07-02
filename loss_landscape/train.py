"""Training script for experiments defined via YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from loss_landscape.utils.config import load_config
from importlib import import_module
import secrets


def seed_everything(seed: int):
    import os, numpy as np
    secrets.SystemRandom().seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(cfg):
    if cfg.data.name.upper() == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(cfg.data.root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(cfg.data.root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset {cfg.data.name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_model(cfg):
    module = import_module(f"loss_landscape.models.{cfg.model.name}")
    return module.build_model(cfg)


def build_optimizer(cfg, model):
    # Optimizer
    if cfg.optimizer.name.upper() == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=getattr(cfg.optimizer, "momentum", 0.0),
            weight_decay=getattr(cfg.optimizer, "weight_decay", 0.0),
        )
    elif cfg.optimizer.name.upper() == "ADAM":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=getattr(cfg.optimizer, "weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer {cfg.optimizer.name}")

    # Scheduler (optional)
    if cfg.scheduler.name.upper() == "STEPLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )
    else:
        scheduler = None
    return optimizer, scheduler


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, log_interval):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    prog = tqdm(enumerate(loader), total=len(loader), ncols=100, leave=False)
    for batch_idx, (data, target) in prog:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(output.detach(), target)

        if (batch_idx + 1) % log_interval == 0:
            prog.set_description(
                f"Iter {batch_idx+1}/{len(loader)} loss={(running_loss/(batch_idx+1)):.4f} acc={(running_acc/(batch_idx+1))*100:.2f}%"
            )

    return running_loss / len(loader), running_acc / len(loader)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss_total = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_total += criterion(output, target).item()
            acc_total += accuracy(output, target)

    return loss_total / len(loader), acc_total / len(loader)


def save_checkpoint(model, optimizer, epoch, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Train a model and save checkpoints for loss landscape analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer, scheduler = build_optimizer(cfg, model)

    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, cfg.train.log_interval
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:2d}: train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

        # checkpoint
        if epoch % cfg.train.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, cfg, cfg.output_dir)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.2f} min. Best val acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main() 
