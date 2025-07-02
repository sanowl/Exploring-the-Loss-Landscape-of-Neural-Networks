# Exploring the Loss Landscape of Neural Networks

This repository contains a **minimal yet extensible** framework for studying the geometry of neural-network loss functions.  It lets you

1. **Train** a small model (e.g. LeNet on MNIST)
2. **Checkpoint** weights and hyper-parameters
3. **Probe** the loss landscape with second-order tools (Hessian eigen-analysis)
4. **Visualize** 1-D and 2-D slices of the loss surface

---

## Quick Start

```bash
# 1. Create a fresh Python environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train LeNet on MNIST (single GPU or CPU)
python -m loss_landscape.train --config configs/mnist_lenet.yaml

# 4. Compute top-20 Hessian eigenvalues at epoch 5
python -m loss_landscape.analysis.hessian \
    --ckpt runs/mnist_lenet/epoch_5.pt \
    --num_eigs 20

# 5. Plot a 2-D slice (random directions)
python -m loss_landscape.analysis.slice \
    --ckpt runs/mnist_lenet/epoch_5.pt ‑-plot 3d
```

---

## Project Layout

```
configs/                # YAML experiment configs
loss_landscape/
    __init__.py
    train.py            # Training & checkpointing CLI
    utils/
        config.py       # YAML loader + dot-dict
    models/
        lenet.py        # Example architecture
    analysis/
        hessian.py      # Hessian eigen-analysis CLI
        slice.py        # 1-D/2-D loss surface probes
runs/                   # Checkpoints & logs go here (auto-created)
```

---

## Extending

1. **Add a new model**: drop a file under `loss_landscape/models/` that exposes `build_model(cfg)`.
2. **Change dataset**: edit or create a YAML under `configs/`.
3. **Swap optimiser**: update the `optimizer:` block in the YAML.

---

## Citation / References

Built with inspiration from:

* Li et al., *Visualizing the Loss Landscape of Neural Nets* (NeurIPS 2018)
* Garipov et al., *Mode Connectivity and the Loss Landscape of Neural Networks* (ICML 2018)
* PyHessian library (Gholami et al.) 