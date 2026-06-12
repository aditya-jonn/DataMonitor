"""Rebuild the master results table (Table 3) from per-run results.csv files
and render it in the terminal.

Per-run files under results/<bszB_metric_seedS>/results.csv are the source of
truth (one row per method, written by ood_detection.py). This script
concatenates them into one CSV (full rebuild + atomic replace, safe to run at
any time, even while evals are still running) and then prints the table:
rich-formatted if 'rich' is installed, plain text otherwise. Pass --quiet to
write the CSV without the terminal rendering.
"""
import argparse
import glob
import json
import os

import pandas as pd

STATS = ("Accuracy", "Specificity", "Sensitivity")


def load_master():
    paths = sorted(glob.glob("results/*/results.csv"))
    if not paths:
        return None, [], None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.sort_values(["Batch Size", "Seed", "Metric", "Method"]).reset_index(drop=True)

    master = "results/ood_bootstrap.csv"
    try:
        with open("cfg.json") as f:
            master = json.load(f).get("table_path", master)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return df, paths, master


def write_master(df, master):
    # PID-unique tmp: every pipeline runs a merge at the end of its eval stage,
    # so concurrent merges are normal. A shared tmp name lets one process
    # rename the file away before its sibling's os.replace (FileNotFoundError);
    # a per-process name makes each rebuild independent, last finisher wins.
    tmp = f"{master}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, master)  # atomic rename: never a partial master


def _ci(row, stat):
    """Fold Mean/LCL/UCL columns into one compact cell: 'mean [lcl, ucl]'."""
    return (f"{row[f'Mean {stat}']:.3f} "
            f"[{row[f'LCL {stat}']:.3f}, {row[f'UCL {stat}']:.3f}]")


def render(df):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        out = df[["Batch Size", "Seed", "Metric", "Method"]].copy()
        for s in STATS:
            out[s] = df.apply(lambda r, s=s: _ci(r, s), axis=1)
        print("\n" + out.to_string(index=False))
        print("\n(install 'rich' in the venv for the formatted view)")
        return

    # bracket-bearing strings go through Text (literal, no markup parsing)
    table = Table(
        title=f"OOD Detection - Table 3 ({len(df)} rows)",
        caption=Text("cells are mean [bootstrap LCL, UCL] - bold green = best mean accuracy within a run key"),
    )
    for col in ("Bsz", "Seed", "Metric", "Method"):
        table.add_column(col, no_wrap=True)
    for col in STATS:
        table.add_column(col, justify="right", no_wrap=True)

    # one section per run key (batch size, seed); blank repeated key cells
    for (bsz, seed), g in df.groupby(["Batch Size", "Seed"], sort=True):
        best = g["Mean Accuracy"].idxmax()
        first = True
        for idx, row in g.iterrows():
            acc = Text(_ci(row, "Accuracy"))
            if idx == best:
                acc.stylize("bold green")
            table.add_row(
                str(bsz) if first else "",
                str(seed) if first else "",
                str(row["Metric"]),
                str(row["Method"]),
                acc,
                Text(_ci(row, "Specificity")),
                Text(_ci(row, "Sensitivity")),
            )
            first = False
        table.add_section()

    console = Console()
    if not console.is_terminal:          # logs / pipes: don't squeeze to 80 cols
        console = Console(width=120)
    console.print(table)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="write the master CSV but skip the terminal table")
    args = ap.parse_args()

    df, paths, master = load_master()
    if df is None:
        print("No per-run files under results/*/results.csv - nothing to merge.")
        return
    write_master(df, master)
    print(f"Merged {len(paths)} per-run files ({len(df)} rows) -> {master}")
    if not args.quiet:
        render(df)


if __name__ == "__main__":
    main()