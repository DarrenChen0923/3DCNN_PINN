
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-Focused Evaluation for CNN3D / PINN / Fusion
=====================================================
Use previously saved *best_model.pth* checkpoints to demonstrate PINN effectiveness near boundaries.

Examples
--------
python evaluate_boundary.py --grid_size 20 --cnn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_20mm/architecture_ablation/CNN3D_Only/best_model.pth --pinn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_20mm/architecture_ablation/PINN_Only/best_model.pth --fusion_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_20mm/architecture_ablation/CNN3D_PINN_Fusion/best_model.pth --batch_size 32 --seed 42 --out_dir boundary_eval_20mm

python evaluate_boundary.py --grid_size 15 --cnn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_15mm/architecture_ablation/CNN3D_Only/best_model.pth --pinn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_15mm/architecture_ablation/PINN_Only/best_model.pth --fusion_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_15mm/architecture_ablation/CNN3D_PINN_Fusion/best_model.pth --batch_size 32 --seed 42 --out_dir boundary_eval_15mm

python evaluate_boundary.py --grid_size 10 --cnn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_10mm/architecture_ablation/CNN3D_Only/best_model.pth --pinn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_10mm/architecture_ablation/PINN_Only/best_model.pth --fusion_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_10mm/architecture_ablation/CNN3D_PINN_Fusion/best_model.pth --batch_size 32 --seed 42 --out_dir boundary_eval_10mm

python evaluate_boundary.py --grid_size 5 --cnn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_5mm/architecture_ablation/CNN3D_Only/best_model.pth --pinn_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_5mm/architecture_ablation/PINN_Only/best_model.pth --fusion_path C:/Users/DuChe/Documents/3cdnnpinn/3DCNN_PINN/ablation_experiments/ablation_experiments_5mm/architecture_ablation/CNN3D_PINN_Fusion/best_model.pth --batch_size 32 --seed 42 --out_dir boundary_eval_5mm

Outputs
-------
- boundary_eval_20mm/summary.json
- boundary_eval_20mm/bucket_metrics.csv
- boundary_eval_20mm/residual_vs_boundary_scatter_*.png
- boundary_eval_20mm/bucket_mae_bar_*.png
- boundary_eval_20mm/pred_target_hist_low_boundary_*.png
- boundary_eval_20mm/per_sample_metrics.csv
- boundary_eval_20mm/preds_*.npy, targets.npy, boundary_scores.npy
"""

import os
import json
import argparse
import numpy as np
import torch

from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from ablation_models import CNN3D_Only_Model, PINN_Only_Model, Baseline_MLP_Model
from models import CNN3D_PINN_Model
from trainer import evaluate_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_KEYS = ["cnn", "pinn", "fusion"]


def parse_args():
    p = argparse.ArgumentParser(description="Boundary-focused evaluation using saved checkpoints")
    p.add_argument("--grid_size", type=int, required=True, help="Grid size in mm (e.g., 5/10/15/20)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory. Defaults to boundary_eval_{grid_size}mm_{timestamp}")
    # model paths
    p.add_argument("--cnn_path", type=str, required=False, help="Path to CNN3D_Only best_model.pth")
    p.add_argument("--pinn_path", type=str, required=False, help="Path to PINN_Only best_model.pth")
    p.add_argument("--fusion_path", type=str, required=False, help="Path to CNN3D_PINN_Fusion best_model.pth")
    # bucketing
    p.add_argument("--num_buckets", type=int, default=5, help="Number of buckets by boundary score (quantiles)")
    p.add_argument("--low_boundary_quantile", type=float, default=0.2, help="Quantile cutoff for 'low-boundary' region")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device_from_arg(arg: str):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_models():
    return {
        "cnn": CNN3D_Only_Model(),
        "pinn": PINN_Only_Model(),
        "fusion": CNN3D_PINN_Model(),
    }


def load_weights(models, paths, device):
    loaded = {}
    for k in MODEL_KEYS:
        path = paths.get(k)
        if path is None:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"{k} checkpoint not found: {path}")
        models[k].load_state_dict(torch.load(path, map_location=device))
        models[k].to(device)
        models[k].eval()
        loaded[k] = True
    return loaded


def boundary_score_from_sample(sample_flat_orig9):
    """Compute boundary score = mean absolute value at four corners (indices 0,2,6,8) on ORIGINAL scale."""
    indices = [0, 2, 6, 8]
    return float(np.mean(np.abs(sample_flat_orig9[indices])))


@torch.no_grad()
def forward_collect(model, loader, device="cpu"):
    """Run model on loader and return per-sample predictions and targets (both normalized scale)."""
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


def compute_metrics(preds, targs):
    mae = float(np.mean(np.abs(preds - targs)))
    mse = float(np.mean((preds - targs) ** 2))
    rmse = float(np.sqrt(mse))
    # r2
    ss_tot = np.sum((targs - np.mean(targs)) ** 2)
    ss_res = np.sum((targs - preds) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_residual_scatter(boundary, residuals, name, save_dir):
    plt.figure(figsize=(7,5))
    plt.scatter(boundary, residuals, s=8, alpha=0.7)
    plt.xlabel("Boundary score (|corner mean|, original scale)")
    plt.ylabel("Residual (pred - target) [normalized]")
    plt.title(f"Residual vs Boundary score - {name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"residual_vs_boundary_scatter_{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_bucket_mae(bucket_edges, bucket_mae_dict, save_dir):
    xs = [f"[{bucket_edges[i]:.2f},{bucket_edges[i+1]:.2f})" for i in range(len(bucket_edges)-1)]
    for name, maes in bucket_mae_dict.items():
        plt.figure(figsize=(8,5))
        plt.bar(range(len(maes)), maes)
        plt.xticks(range(len(maes)), xs, rotation=30, ha="right")
        plt.ylabel("MAE (normalized)")
        plt.title(f"MAE by Boundary-score bucket - {name}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"bucket_mae_bar_{name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_low_boundary_hist(preds_norm, targs_norm, y_scaler, name, save_dir):
    # inverse transform for readability
    preds_orig = y_scaler.inverse_transform(preds_norm.reshape(-1,1)).reshape(-1)
    targs_orig = y_scaler.inverse_transform(targs_norm.reshape(-1,1)).reshape(-1)
    plt.figure(figsize=(7,5))
    plt.hist(preds_orig, bins=30, alpha=0.7, label="Pred (orig)")
    plt.hist(targs_orig, bins=30, alpha=0.5, label="Target (orig)")
    plt.xlabel("Springback error (original scale)")
    plt.ylabel("Count")
    plt.title(f"Distribution @ Low-boundary subset - {name}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"pred_target_hist_low_boundary_{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = _device_from_arg(args.device)

    # Prepare out dir
    out_dir = args.out_dir or f"boundary_eval_{args.grid_size}mm"
    os.makedirs(out_dir, exist_ok=True)

    # Load data & preprocess
    X, y = load_data_by_grid_size(args.grid_size)
    data = preprocess_data(X, y, test_size=0.1, val_size=0.1, random_state=args.seed)
    loaders = create_data_loaders(data, batch_size=args.batch_size)

    # Build mapping from test set back to original-scale boundary scores
    # NOTE: data['X_test'] are standardized. We inverse-transform to original scale first.
    X_test_scaled = data['X_test']  # shape [N, 9]
    X_test_orig = data['X_scaler'].inverse_transform(X_test_scaled)  # original scale
    boundary_scores = np.array([boundary_score_from_sample(x) for x in X_test_orig])  # shape [N]

    # Assemble models
    models = build_models()
    paths = {"cnn": args.cnn_path, "pinn": args.pinn_path, "fusion": args.fusion_path}
    loaded = load_weights(models, paths, device)
    if not loaded:
        raise RuntimeError("No checkpoints loaded. Provide at least one of --cnn_path/--pinn_path/--fusion_path.")

    # Evaluate per model
    results = {}
    per_sample_rows = []  # for CSV
    residual_scatter_paths = []
    bucket_mae_dict = {}
    bucket_edges = np.quantile(boundary_scores, np.linspace(0, 1, args.num_buckets+1))

    # For low-boundary subset distribution figure
    low_q = args.low_boundary_quantile
    low_thr = np.quantile(boundary_scores, low_q)
    low_mask = boundary_scores <= low_thr

    # Common targets from the loader (we will run once with any loaded model to capture targets)
    # Use fusion if available, else cnn, else pinn, purely to reuse forward pass loop
    key_for_targets = "fusion" if "fusion" in loaded else ("cnn" if "cnn" in loaded else "pinn")
    _, targs_norm = forward_collect(models[key_for_targets], loaders["test_loader"], device=device)

    # Save targets and boundary_scores
    np.save(os.path.join(out_dir, "targets.npy"), targs_norm)
    np.save(os.path.join(out_dir, "boundary_scores.npy"), boundary_scores)

    for name in MODEL_KEYS:
        if name not in loaded:
            continue
        preds_norm, targs_norm2 = forward_collect(models[name], loaders["test_loader"], device=device)
        assert np.allclose(targs_norm, targs_norm2), "Targets mismatch across forward passes"

        # Save preds
        np.save(os.path.join(out_dir, f"preds_{name}.npy"), preds_norm)

        # Global metrics
        metrics_global = compute_metrics(preds_norm, targs_norm)

        # Residual vs boundary scatter
        residuals = preds_norm - targs_norm
        scatter_path = plot_residual_scatter(boundary_scores, residuals, name, out_dir)
        residual_scatter_paths.append(scatter_path)

        # Bucketed metrics
        maes = []
        for i in range(len(bucket_edges)-1):
            lo, hi = bucket_edges[i], bucket_edges[i+1]
            mask = (boundary_scores >= lo) & (boundary_scores < hi) if i < len(bucket_edges)-2 else (boundary_scores >= lo) & (boundary_scores <= hi)
            if np.any(mask):
                m = compute_metrics(preds_norm[mask], targs_norm[mask])
                maes.append(m["mae"])
                # record per-sample rows for optional deep analysis
                for idx in np.where(mask)[0]:
                    per_sample_rows.append({
                        "model": name,
                        "index": int(idx),
                        "boundary_bucket_lo": float(lo),
                        "boundary_bucket_hi": float(hi),
                        "boundary_score": float(boundary_scores[idx]),
                        "pred_norm": float(preds_norm[idx]),
                        "targ_norm": float(targs_norm[idx]),
                        "residual_norm": float(preds_norm[idx] - targs_norm[idx])
                    })
            else:
                maes.append(float("nan"))
        bucket_mae_dict[name] = maes

        # Low-boundary subset distribution
        _ = plot_low_boundary_hist(preds_norm[low_mask], targs_norm[low_mask], data["y_scaler"], name, out_dir)

        # Save model-level summary
        results[name] = {
            "global": metrics_global,
            "low_boundary_quantile": low_q,
            "low_boundary_threshold": float(low_thr),
            "num_low_boundary_samples": int(np.sum(low_mask))
        }

    # Save bucket metrics CSV
    import csv
    csv_path = os.path.join(out_dir, "bucket_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["bucket_idx", "lo", "hi"] + [f"{k}_mae" for k in bucket_mae_dict.keys()]
        writer.writerow(header)
        for i in range(len(bucket_edges)-1):
            row = [i, float(bucket_edges[i]), float(bucket_edges[i+1])]
            for k in bucket_mae_dict.keys():
                val = bucket_mae_dict[k][i]
                row.append(float(val) if val==val else None)
            writer.writerow(row)

    # Save per-sample CSV
    import pandas as pd
    df = pd.DataFrame(per_sample_rows)
    df.to_csv(os.path.join(out_dir, "per_sample_metrics.csv"), index=False)

    # Plot bucket bar per model
    plot_bucket_mae(bucket_edges, bucket_mae_dict, out_dir)

    # Save summary.json
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "grid_size": args.grid_size,
            "num_test_samples": int(len(X_test_orig)),
            "num_buckets": args.num_buckets,
            "bucket_edges": [float(x) for x in bucket_edges],
            "models_evaluated": list(bucket_mae_dict.keys()),
            "results": results
        }, f, indent=4, ensure_ascii=False)

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
