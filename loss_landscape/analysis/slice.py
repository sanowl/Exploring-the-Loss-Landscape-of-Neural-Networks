"""Plot 1-D or 2-D slices of the loss landscape around a given checkpoint.

Example (2-D random directions)::

    python -m loss_landscape.analysis.slice \
        --ckpt runs/mnist_lenet/epoch_5.pt \
        --range 1.0 --num_points 21 --plot 3d
"""
from __future__ import annotations

import argparse
from pathlib import Path
import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np
from importlib import import_module

from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def build_dataset(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if cfg.data.name.upper() == "MNIST":
        test_ds = datasets.MNIST(cfg.data.root, train=False, download=True, transform=transform)
    else:
        raise ValueError("Only MNIST supported for now in slice script")
    loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)
    return loader


def get_random_directions(model):
    # Return two random, L2-normalized direction vectors with the same shape as model parameters
    directions = []
    for _ in range(2):
        vec = []
        for p in model.parameters():
            rand_dir = torch.randn_like(p)
            vec.append(rand_dir)
        directions.append(vec)
    # normalize directions
    for d in directions:
        norm = torch.sqrt(sum([(v ** 2).sum() for v in d]))
        for v in d:
            v /= norm
    return directions


def model_forward_loss(model, criterion, data_loader, device):
    # Evaluate loss over the entire loader
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            count += inputs.size(0)
    return total_loss / count


def add_direction(model, base_params, directions, coeffs):
    # Set model parameters to base + a*d1 + b*d2 (in-place)
    for p, base, d1, d2 in zip(model.parameters(), base_params, directions[0], directions[1]):
        p.data = base + coeffs[0] * d1 + coeffs[1] * d2


def slice_2d(model, base_params, directions, alphas, betas, data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    Z = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            add_direction(model, base_params, directions, (a, b))
            Z[i, j] = model_forward_loss(model, criterion, data_loader, device)
    # restore original params
    for p, base in zip(model.parameters(), base_params):
        p.data = base
    return Z


def main():
    parser = argparse.ArgumentParser(description="Plot 1-D/2-D loss landscape slices")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--range", type=float, default=1.0, help="Range of alpha/beta (±range)")
    parser.add_argument("--num_points", type=int, default=21, help="Samples per axis")
    parser.add_argument("--plot", choices=["contour", "3d"], default="contour", help="Plot type for 2D slice")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    if not hasattr(cfg, "__dict__"):
        from types import SimpleNamespace
        cfg = SimpleNamespace(**cfg)

    module = import_module(f"loss_landscape.models.{cfg.model.name}")
    model = module.build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_loader = build_dataset(cfg)

    # Prepare base parameters and directions
    base_params = [p.clone().detach() for p in model.parameters()]
    directions = get_random_directions(model)

    alphas = np.linspace(-args.range, args.range, args.num_points)
    betas = np.linspace(-args.range, args.range, args.num_points)

    print("Computing loss surface...")
    Z = slice_2d(model, base_params, directions, alphas, betas, data_loader, device)

    # Plot
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    if args.plot == "contour":
        plt.figure(figsize=(6, 5))
        cp = plt.contourf(A, B, Z, levels=50, cmap="viridis")
        plt.colorbar(cp)
        plt.xlabel("alpha (dir 1)")
        plt.ylabel("beta (dir 2)")
        plt.title("2-D Loss Landscape Slice")
        plt.tight_layout()
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, antialiased=False)
        ax.set_xlabel("alpha (dir 1)")
        ax.set_ylabel("beta (dir 2)")
        ax.set_zlabel("Loss")
        plt.title("3-D Loss Landscape Slice")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main() 