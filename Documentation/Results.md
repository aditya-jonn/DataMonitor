# Results


## Reproducing the master table

The master table is built by `merge_results.py` from the per-run `results.csv` files that `ood_detection.py` writes.

```bash
python merge_results.py            # rebuild the master CSV + print the table
python merge_results.py --quiet    # rebuild the CSV only, no rendering
```

It globs every `results/*/results.csv`, concatenates, sorts by `[Batch Size, Seed, Metric, Method]`, and writes the master CSV atomically (PID-unique temp file, then `os.replace`), so it is safe to run while evaluations are in flight (last finisher wins; the path comes from `cfg.json`'s `table_path`, default `results/ood_bootstrap.csv`). It then renders the table grouped one section per `(batch size, seed)`, best mean accuracy per section highlighted. Each cell is `mean [LCL, UCL]`, where the bounds are `mean ± 1 std` of the bootstrap, not 95% intervals. The per-run files are the source of truth; the master is a derived view, rebuildable at any time.

---

## The seed × batch-size figure

Trains each extractor across four batch sizes `{16, 32, 128, 256}` and ten seeds `1001–1010`, yielding 40 run keys. Each key is evaluated under all four metrics for all three methods (12 rows per key, 480 rows, a complete grid). `plot_seed_batch.py` renders the analysis figure (paper Figure 2) from the master CSV: one panel per method, mean detection accuracy versus batch size, with a ±1σ band across the random seeds.

```bash
python plot_seed_batch.py                                  # defaults
python plot_seed_batch.py --csv results/ood_bootstrap.csv --out seed_batch_analysis.png
python plot_seed_batch.py --metric "Mean Sensitivity"      # any Mean* column
python plot_seed_batch.py --print-summary                  # also dump per-cell stats
```

Two modelling choices follow from the design: `mahalanobis-solve` is dropped (identical to `mahalanobis`), and `mahalanobis-pinv` is kept and drawn dashed (it differs only in the covariance estimator, so it shows the estimator's effect on accuracy).
