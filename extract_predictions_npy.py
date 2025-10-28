
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper: extract predictions/targets to .npy from any saved checkpoint.

Example:
python extract_predictions_npy.py --grid_size 20 --model_type fusion --ckpt_path results/grid_20mm_.../CNN3D_PINN_Fusion/best_model.pth --out_dir npy_dump_20mm_fusion

model_type ∈ {"cnn","pinn","fusion"}.
"""

import os
import argparse
import numpy as np
import torch

from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from ablation_models import CNN3D_Only_Model, PINN_Only_Model
from models import CNN3D_PINN_Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, required=True)
    p.add_argument("--model_type", type=str, choices=["cnn","pinn","fusion"], required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def set_seed(seed: int):
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(kind: str):
    if kind == "cnn":
        return CNN3D_Only_Model()
    if kind == "pinn":
        return PINN_Only_Model()
    return CNN3D_PINN_Model()


@torch.no_grad()
def forward_collect(model, loader, device="cpu"):
    preds, targs = [], []
    for batch in loader:
        x = batch["point_series"].to(device)
        y = batch["error"].to(device)
        out, _, _ = model(x)
        preds.append(out.cpu().numpy())
        targs.append(y.cpu().numpy())
    preds = np.vstack(preds).reshape(-1)
    targs = np.vstack(targs).reshape(-1)
    return preds, targs


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu"))

    os.makedirs(args.out_dir, exist_ok=True)

    # data
    X, y = load_data_by_grid_size(args.grid_size)
    data = preprocess_data(X, y, test_size=0.1, val_size=0.1, random_state=args.seed)
    loaders = create_data_loaders(data, batch_size=args.batch_size)

    # model
    model = build_model(args.model_type)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(args.ckpt_path)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.to(device).eval()

    preds, targs = forward_collect(model, loaders["test_loader"], device=device)

    np.save(os.path.join(args.out_dir, "preds.npy"), preds)
    np.save(os.path.join(args.out_dir, "targets.npy"), targs)
    print(f"Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
