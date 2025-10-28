
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate boundary-evaluation results across grid sizes")
    p.add_argument("--dirs", nargs="+", required=True, help="List of evaluate_boundary output folders")
    p.add_argument("--out_dir", type=str, default="boundary_eval_aggregate")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for CI on low-boundary improvement")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_one_dir(d: str):
    d = Path(d)
    with open(d / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    bucket = pd.read_csv(d / "bucket_metrics.csv")
    per_sample = pd.read_csv(d / "per_sample_metrics.csv")
    summary["dir"] = str(d)
    return summary, bucket, per_sample


def longform_bucket(bucket: pd.DataFrame, models, grid_label: str):
    rows = []
    for _, r in bucket.iterrows():
        for m in models:
            key = f"{m}_mae"
            if key in bucket.columns:
                rows.append({
                    "grid": grid_label,
                    "bucket_idx": int(r["bucket_idx"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                    "model": m,
                    "mae": float(r[key]) if pd.notna(r[key]) else np.nan
                })
    return pd.DataFrame(rows)


def mae_low_boundary(per_sample: pd.DataFrame, model: str, low_thr: float) -> float:
    """Compute MAE in low-boundary subset for a single model.
    IMPORTANT: low-boundary mask must be computed AFTER filtering by model to avoid length mismatch.
    """
    df_m = per_sample[per_sample["model"] == model].copy()
    if df_m.empty:
        return np.nan
    mask_m = df_m["boundary_score"].values <= low_thr
    df_low = df_m[mask_m]
    if df_low.empty:
        return np.nan
    return float(np.mean(np.abs(df_low["residual_norm"].values)))


def bootstrap_ci_improvement(per_sample: pd.DataFrame, low_thr: float, models, B=2000, seed=42):
    """Bootstrap CI for (CNN - Fusion) / CNN on low-boundary subset."""
    rng = np.random.default_rng(seed)

    df_cnn = per_sample[(per_sample["model"] == "cnn") & (per_sample["boundary_score"].values <= low_thr)]
    df_fus = per_sample[(per_sample["model"] == "fusion") & (per_sample["boundary_score"].values <= low_thr)]
    if df_cnn.empty or df_fus.empty:
        return {}

    a = np.abs(df_fus["residual_norm"].values)
    b = np.abs(df_cnn["residual_norm"].values)
    n_a, n_b = len(a), len(b)
    boot = []
    for _ in range(B):
        ia = rng.integers(0, n_a, n_a)
        ib = rng.integers(0, n_b, n_b)
        mae_f = np.mean(a[ia])
        mae_c = np.mean(b[ib])
        imp = (mae_c - mae_f) / mae_c if mae_c != 0 else 0.0
        boot.append(imp)
    arr = np.array(boot)
    return {
        "fusion_vs_cnn_low_boundary_bootstrap": {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "n_low": int(len(df_cnn) + len(df_fus))  # total low-boundary samples counted across two models
        }
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    long_rows = []
    low_summary_rows = []

    for d in args.dirs:
        summary, bucket, per_sample = load_one_dir(d)
        grid = f'{summary.get("grid_size", "NA")}mm'
        models = [m for m in ["cnn","pinn","fusion"] if m in summary.get("models_evaluated", ["cnn","pinn","fusion"])]

        lf = longform_bucket(bucket, models, grid)
        long_rows.append(lf)

        # low-boundary threshold taken from any model's entry (same threshold across models in the run)
        first_model = models[0]
        low_thr = summary["results"][first_model]["low_boundary_threshold"]

        row = {"grid": grid, "low_boundary_threshold": low_thr}
        for m in models:
            row[f"mae_low_{m}"] = mae_low_boundary(per_sample, m, low_thr)

        if "fusion" in models and "cnn" in models:
            mae_c = row["mae_low_cnn"]
            mae_f = row["mae_low_fusion"]
            if (mae_c is not None) and (not np.isnan(mae_c)) and mae_c != 0:
                row["improve_fusion_vs_cnn_low"] = (mae_c - mae_f) / mae_c
            else:
                row["improve_fusion_vs_cnn_low"] = np.nan

            boot = bootstrap_ci_improvement(per_sample, low_thr, models, B=args.bootstrap, seed=args.seed)
            row.update(boot.get("fusion_vs_cnn_low_boundary_bootstrap", {}))

        low_summary_rows.append(row)

        # per-grid plot: MAE vs bucket
        wide = lf.pivot_table(index="bucket_idx", columns="model", values="mae", aggfunc="first").sort_index()
        plt.figure(figsize=(7,5))
        for m in [x for x in ["cnn","pinn","fusion"] if x in wide.columns]:
            plt.plot(wide.index.values, wide[m].values, marker="o", label=m)
        plt.xlabel("Boundary-score bucket index (low → high)")
        plt.ylabel("MAE (normalized)")
        plt.title(f"MAE vs Boundary bucket — {grid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"mae_vs_bucket_{grid}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Save combined long-form buckets
    df_long = pd.concat(long_rows, ignore_index=True)
    df_long.to_csv(os.path.join(args.out_dir, "combined_bucket_metrics.csv"), index=False)

    # Save low-boundary summary
    df_low = pd.DataFrame(low_summary_rows)
    cols = ["grid", "low_boundary_threshold",
            "mae_low_cnn", "mae_low_pinn", "mae_low_fusion",
            "improve_fusion_vs_cnn_low",
            "mean", "ci95_low", "ci95_high", "n_low"]
    for c in cols:
        if c not in df_low.columns:
            df_low[c] = np.nan
    df_low = df_low[cols]
    df_low.to_csv(os.path.join(args.out_dir, "low_boundary_summary.csv"), index=False)

    # Bar plot: improvement with CI
    if df_low["improve_fusion_vs_cnn_low"].notna().any():
        plt.figure(figsize=(7,5))
        x = np.arange(len(df_low))
        y = df_low["improve_fusion_vs_cnn_low"].values * 100.0
        plt.bar(x, y)
        if df_low["ci95_low"].notna().any():
            ylow = (df_low["ci95_low"].values * 100.0)
            yhigh = (df_low["ci95_high"].values * 100.0)
            err_low = y - ylow
            err_high = yhigh - y
            plt.errorbar(x, y, yerr=[err_low, err_high], fmt="none", capsize=4)
        plt.xticks(x, df_low["grid"].values, rotation=0)
        plt.ylabel("Fusion vs CNN — Low-boundary MAE Improvement (%)")
        plt.title("Low-boundary improvement across grid sizes")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fusion_vs_cnn_improvement_low_boundary.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # LaTeX table
    def fmt(x):
        return "—" if pd.isna(x) else f"{x:.3f}"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Grid & MAE$_{\text{low}}$(CNN) & MAE$_{\text{low}}$(PINN) & MAE$_{\text{low}}$(Fusion) & $\Delta$(Fusion vs CNN) \\")
    lines.append(r"\midrule")
    for _, r in df_low.iterrows():
        imp = r["improve_fusion_vs_cnn_low"]
        imp_str = "—"
        if not pd.isna(imp):
            if not pd.isna(r["ci95_low"]):
                imp_str = f"{imp*100:.1f}\\% [ {r['ci95_low']*100:.1f}, {r['ci95_high']*100:.1f} ]"
            else:
                imp_str = f"{imp*100:.1f}\\%"
        lines.append(f"{r['grid']} & {fmt(r['mae_low_cnn'])} & {fmt(r['mae_low_pinn'])} & {fmt(r['mae_low_fusion'])} & {imp_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Low-boundary MAE across grid sizes (lower is better). $\Delta$ shows Fusion over CNN improvement with 95\% bootstrap CI.}")
    lines.append(r"\label{tab:low_boundary_across_grids}")
    lines.append(r"\end{table}")
    with open(os.path.join(args.out_dir, "boundary_across_grids.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Done. Outputs in {args.out_dir}")
    print(f"- combined_bucket_metrics.csv")
    print(f"- low_boundary_summary.csv")
    print(f"- fusion_vs_cnn_improvement_low_boundary.png")
    print(f"- mae_vs_bucket_[GRID].png")
    print(f"- boundary_across_grids.tex")

if __name__ == "__main__":
    main()
