#!/usr/bin/env python3
"""Seed x batch-size analysis figure for the OOD-detection sweep.

Reads the master results table (results/ood_bootstrap.csv, the file produced by
merge_results.py) and renders one panel per feature method showing mean
detection accuracy versus batch size, with a +/-1 sigma band taken ACROSS the
random seeds. The band is the seed-to-seed noise floor; a batch-size trend is
meaningful only where it rises above that band.

Two modelling choices, both grounded in the pipeline's design:

  * 'mahalanobis-solve' is dropped. It shares the Ledoit-Wolf covariance with
    'mahalanobis' and (verified on this data) is bit-identical to it, so it is
    the same metric by a different procedure -- not an independent series.
  * 'mahalanobis-pinv' is kept and drawn dashed: it differs only in the
    covariance estimator (raw pseudo-inverse vs shrinkage) and is shown so the
    estimator's effect on accuracy is visible.

Usage:
    python plot_seed_batch.py                          # defaults below
    python plot_seed_batch.py --csv results/ood_bootstrap.csv \
                              --out seed_batch_analysis.png
    python plot_seed_batch.py --metric "Mean Sensitivity"   # any Mean* column
    python plot_seed_batch.py --print-summary          # also dump per-cell stats
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: write a file, never open a window
import matplotlib.pyplot as plt

# Drawing order and styling. maha-pinv dashed so it reads as the baseline.
METRICS = ["cosine", "maha (=solve)", "maha-pinv"]
COLORS = {"cosine": "#1f77b4", "maha (=solve)": "#d62728", "maha-pinv": "#ff7f0e"}
METHODS = ["autoencoder", "cnn", "ctr"]
RENAME = {"mahalanobis": "maha (=solve)", "mahalanobis-pinv": "maha-pinv"}


def load(csv_path):
    """Load the master table, drop the duplicate solve metric, tidy names."""
    df = pd.read_csv(csv_path)
    df = df[df["Metric"] != "mahalanobis-solve"].copy()
    df["Metric"] = df["Metric"].replace(RENAME)
    return df


def summarize(df, value_col):
    """Seed-mean and seed-std of value_col per (Method, Metric, Batch Size)."""
    g = df.groupby(["Method", "Metric", "Batch Size"])[value_col]
    out = g.agg(seed_mean="mean", seed_std="std", n="count").reset_index()
    return out


def make_figure(df, value_col, out_path):
    batch_sizes = sorted(df["Batch Size"].unique())
    x = np.arange(len(batch_sizes))            # categorical: batch sizes are not linear
    methods = [m for m in METHODS if m in df["Method"].unique()]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        for metric in METRICS:
            sub = df[(df["Method"] == method) & (df["Metric"] == metric)]
            if sub.empty:
                continue
            m = sub.groupby("Batch Size")[value_col].mean().reindex(batch_sizes)
            s = sub.groupby("Batch Size")[value_col].std().reindex(batch_sizes)
            ls = "--" if metric == "maha-pinv" else "-"
            ax.plot(x, m.values, ls, marker="o", color=COLORS[metric],
                    label=metric, zorder=3)
            ax.fill_between(x, (m - s).values, (m + s).values,
                            color=COLORS[metric], alpha=0.15, zorder=1)
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_title(method)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel("batch size")
        ax.grid(alpha=0.3)

    n_seeds = int(df.groupby(["Method", "Metric", "Batch Size"]).size().max())
    axes[0].set_ylabel(f"{value_col} (\u00b11\u03c3 across {n_seeds} seeds)")
    axes[-1].legend(fontsize=8, loc="lower right")
    seeds = sorted(df["Seed"].unique())
    fig.suptitle(
        f"OOD detection accuracy vs batch size \u2014 seed-mean with noise band "
        f"({len(seeds)} seeds: {seeds[0]}\u2013{seeds[-1]})",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/ood_bootstrap.csv",
                    help="master results CSV (default: results/ood_bootstrap.csv)")
    ap.add_argument("--out", default="seed_batch_analysis.png",
                    help="output image path (default: seed_batch_analysis.png)")
    ap.add_argument("--metric", default="Mean Accuracy",
                    help="which Mean* column to plot (default: Mean Accuracy)")
    ap.add_argument("--print-summary", action="store_true",
                    help="also print the per-cell seed-mean/seed-std table")
    args = ap.parse_args()

    df = load(args.csv)
    if args.print_summary:
        s = summarize(df, args.metric)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(s.to_string(index=False))
    make_figure(df, args.metric, args.out)


if __name__ == "__main__":
    main()
