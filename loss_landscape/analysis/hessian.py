"""Compute Hessian eigenvalues near a checkpoint.

Examples
--------
python -m loss_landscape.analysis.hessian --ckpt runs/mnist_lenet/epoch_5.pt --num_eigs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from importlib import import_module

from loss_landscape.utils.config import load_config

try:
    from pyhessian import hessian as pyhessian
except ImportError:
    pyhessian = None


def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    if not isinstance(cfg, dict):
        # cfg is namespace from SimpleNamespace; convert to dict recursively
        import json, types

        def ns_to_dict(ns):
            if isinstance(ns, types.SimpleNamespace):
                return {k: ns_to_dict(v) for k, v in ns.__dict__.items()}
            else:
                return ns

        cfg = ns_to_dict(cfg)
    cfg = load_config(Path("temp.yaml"))  # dummy override later; just to reuse types
    # Actually easier: load_config again from original config path if stored; fallback.
    return ckpt


def build_dataset(cfg):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if cfg.data.name.upper() == "MNIST":
        test_ds = datasets.MNIST(cfg.data.root, train=False, download=True, transform=transform)
    else:
        raise ValueError("Only MNIST supported for now in Hessian analysis")
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False)
    return test_loader


def main():
    parser = argparse.ArgumentParser(description="Hessian eigen-analysis around a checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num_eigs", type=int, default=20, help="Number of leading eigenvalues to compute")
    parser.add_argument("--use_gpu", action="store_true", help="Force use CUDA if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    # Convert cfg to SimpleNamespace if needed
    if not hasattr(cfg, "__dict__"):
        # assume it's dict-like
        from types import SimpleNamespace
        cfg = SimpleNamespace(**cfg)

    # Rebuild model
    module = import_module(f"loss_landscape.models.{cfg.model.name}")
    model = module.build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Prepare dataset (single batch suffices for Hessian approx)
    test_loader = build_dataset(cfg)
    data_iter = iter(test_loader)
    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device), targets.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    if pyhessian is not None:
        hessian_comp = pyhessian(model, criterion, data=(inputs, targets), cuda=(device.type == "cuda"))
        top_eigs = hessian_comp.eigenvalues(top_n=args.num_eigs)
        print("Top eigenvalues:")
        for idx, eig in enumerate(top_eigs):
            print(f"  λ_{idx+1}: {eig:.6f}")
    else:
        print("PyHessian not installed. Falling back to power iteration (slow).")
        # Flatten parameters
        params = [p for p in model.parameters() if p.requires_grad]

        def hvp(vec):
            # Compute Hessian-vector product using autograd.
            model.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grad_params])
            grad_dot_vec = torch.dot(flat_grad, vec)
            hv = torch.autograd.grad(grad_dot_vec, params, retain_graph=False)
            flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
            return flat_hv

        flat_params = torch.cat([p.reshape(-1) for p in params])
        total_params = flat_params.numel()
        print(f"Total parameters: {total_params}")

        eigvals = []
        vecs = []

        def orthogonalize(v, basis):
            for b in basis:
                v -= torch.dot(v, b) * b
            return v

        for i in range(args.num_eigs):
            vec = torch.randn(total_params, device=device)
            vec = orthogonalize(vec, vecs)
            vec /= vec.norm()

            for _ in range(30):
                hv = hvp(vec)
                # Gram-Schmidt against previous eigenvectors each iteration
                hv = orthogonalize(hv, vecs)
                vec = hv / hv.norm()

            eigval = torch.dot(vec, hvp(vec)).item()
            eigvals.append(eigval)
            vecs.append(vec.detach())
            print(f"  λ_{i+1}: {eigval:.6f}")

    print("Done.")


if __name__ == "__main__":
    main() 