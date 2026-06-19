# Results & Limitations

How to reproduce the paper's headline artifacts, the master detection table ("Table 3") and the seed × batch-size figure ("Figure 2"), how to read the numbers they produce, and the caveats that bound the conclusions.

---

## Reproducing the master table

The master table is built by `merge_results.py` from the per-run `results.csv` files that `ood_detection.py` writes.

```bash
# After one or more pipelines have produced results/*/results.csv:
python merge_results.py            # rebuild the master CSV + print the table
python merge_results.py --quiet    # rebuild the CSV only, no terminal rendering
```

`merge_results.py` **globs** every `results/*/results.csv`, concatenates, and sorts by `[Batch Size, Seed, Metric, Method]`; **writes** the master CSV atomically (PID-unique temp file, then `os.replace`), so it is safe to run while evaluations are in flight and concurrent merges never corrupt the file (last finisher wins; the path comes from `cfg.json`'s `table_path`, default `results/ood_bootstrap.csv`); and **renders** the table (rich-formatted if `rich` is installed, plain otherwise), grouped one section per `(batch size, seed)` run key, the best mean accuracy per section in bold green. Each cell shows `mean [LCL, UCL]`, where the bounds are `mean ± 1 std` of the bootstrap, not 95% intervals. The per-run `results.csv` files are the source of truth; the master is a derived view rebuildable at any time.

---

## Reproducing the seed × batch-size figure

`plot_seed_batch.py` renders the analysis figure (paper Figure 2) from the master CSV: one panel per feature method, mean detection accuracy versus batch size, with a ±1σ band across the random seeds.

```bash
python plot_seed_batch.py                                  # defaults
python plot_seed_batch.py --csv results/ood_bootstrap.csv --out seed_batch_analysis.png
python plot_seed_batch.py --metric "Mean Sensitivity"      # any Mean* column
python plot_seed_batch.py --print-summary                  # also dump per-cell stats
```

Two modelling choices follow from the pipeline's design: `mahalanobis-solve` is dropped (it shares the Ledoit–Wolf covariance with `mahalanobis` and is bit-identical to it on this data, the same metric by a different procedure), and `mahalanobis-pinv` is kept and drawn dashed (it differs only in the covariance estimator, so it shows the estimator's effect on accuracy). The band in each panel is the seed-to-seed noise floor, so a batch-size trend is meaningful only where it rises above that band; a dotted line marks chance (0.5).

---

## The experimental design

The paper trains each extractor across a grid of four batch sizes `{16, 32, 128, 256}` and ten seeds `1001–1010`, yielding 40 run keys. Each key is evaluated under all four metrics for all three methods (12 rows per key, 480 rows total, a complete grid with no missing cells).

The two factors play different roles. The **seed** is a *nuisance* factor: varying it with everything else fixed measures the pipeline's intrinsic run-to-run noise (the "noise floor"). The **batch size** is the *factor of interest*. The analysis is therefore sequential: first establish the seed-induced noise floor, then ask whether any batch-size difference exceeds it. Each cell is summarised as `mean ± σ_seed` over the ten seeds, an ±1σ (≈ 68%) summary from `n = 10` samples, not a 95% interval.

---

## How to read the numbers

### Integrity of the numerical variants

Before interpreting performance, the paper confirms the relationship the metrics should satisfy. `mahalanobis` and `mahalanobis-solve` share the same Ledoit–Wolf covariance and differ only in procedure; on this data they are bit-identical across all 120 cells (max absolute difference in mean accuracy = 0), so they are one metric and `mahalanobis-solve` is dropped from the figures. `mahalanobis-pinv` (identical mathematics, raw pseudo-inverse instead of shrinkage) differs from the shrinkage path by at most 0.018 in mean accuracy (mean difference 0.003): the ill-conditioning is real but, on these trained features, immaterial to detection accuracy. It is kept (dashed) so the near-equivalence is visible rather than assumed.

### The noise floor

For the two methods that detect at all (CNN and contrastive), the typical seed-to-seed standard deviation of accuracy is about one percentage point (median `σ_seed` 0.010 for the CNN, 0.008 for the contrastive encoder). Re-running with a different seed moves accuracy by roughly ±0.01, the bar a batch-size difference must clear to count as real.

The autoencoder is the exception: it sits at chance (≈ 0.51) for every metric and batch size, with a noise floor near zero (`σ_seed ≤ 0.004`). Its reconstruction-latent features do not separate the in-distribution from the off-axis views, so there is no signal for seed or batch size to perturb. It is reported for completeness but excluded from the effect analysis.

### Batch-size effects

The batch-size effect is genuine but lives entirely at the extremes of the range and is method-specific:

- **Contrastive encoder: a small-batch collapse (as predicted by the loss).** From batch size 32 upward the contrastive encoder is flat and strong (cosine accuracy 0.927–0.932, `σ_seed ≤ 0.010`). At batch size 16, cosine accuracy drops to 0.876 and its noise floor explodes to `σ_seed = 0.091`, by far the largest variability in the study (per-seed accuracy ranges 0.684 to 0.955). The sensitivity/specificity decomposition localises the cause: at batch size 16, sensitivity (in-distribution recall, note the inverted-label convention) falls to 0.811 from ≈ 0.99 at larger batches, while specificity rises. The small batch provides too few positives for the contrastive loss to form a tight in-distribution manifold, so the score distribution smears. This is a small-batch failure, not a smooth trend.
- **CNN: a mild large-batch softening, partly an optimisation artifact.** Under Mahalanobis scoring, CNN accuracy declines monotonically with batch size (0.897 → 0.870 from 16 to 256, against a 0.010 noise floor, a real effect). Under cosine scoring, batch size 256 is *unstable* rather than simply lower (`σ_seed = 0.043`). With the learning rate fixed at 0.001 for every configuration, a larger batch means fewer optimiser updates per epoch, so part of this softening is plausibly undertraining rather than a property of batch size as such. The middle of the range (32–128) is stable for the CNN under every metric.

### Metric ranking

Across the working regime the ordering is cosine ≳ Mahalanobis ≈ Mahalanobis-pinv. Cosine is the strongest scorer on these features, and the covariance-estimator choice does not change the ranking.

### Practical takeaway

The seed effect is small and structureless (≈ ±0.01, no dependence on the seed's numeric value). The batch-size effect is real but confined to the endpoints and method-dependent: the contrastive encoder fails below batch size 32, the CNN softens above batch size 128 (entangled with the fixed learning rate), and the autoencoder detects at chance regardless. The defensible operating range is batch size 32–128, where every working method sits in a flat, low-variance band, with the contrastive encoder under cosine scoring the strongest configuration overall.

---

## Study caveats

Three limitations bound the conclusions above:

1. **Small sample for the noise floor.** Every standard deviation is estimated from `n = 10` seeds. The noise-floor estimates are honest but not tight, and all bands are ±1σ (≈ 68%), not 95% intervals.
2. **Learning rate confounded with batch size.** The learning rate is fixed at 0.001 across all batch sizes, so batch size and the number of optimiser steps per epoch co-vary. The CNN's large-batch softening in particular is entangled with optimisation, and attributing it cleanly would need a learning-rate-scaled rerun at batch size 256.
3. **The contrastive collapse leans on one seed.** The batch-size-16 collapse is driven partly by a single poor seed (1006). The elevated variance is itself the finding, but re-extracting that run would confirm the collapse reproduces rather than reflecting a one-off training divergence.

A scope note: the experiment uses a single in-distribution view (axial) against two off-axis views as OOD, on one dataset family (MedMNIST AbdominalCT). The autoencoder's chance-level performance is a finding about *these features on these views*, not a general statement about reconstruction-based OOD detection.

---

## Code-level caveats

- **Upstream modules are not reproduced here.** The base classes (`Model`, `FeatureSpace`), the dataset loader/class, and the two `SPC_Charts/` modules are invoked but originate upstream. They are documented at the interface level; their internals are out of scope.
- **The numerical fail-loud policy is strict by design.** `ood_detection.py` promotes every `RuntimeWarning` to a hard error at import. This is intentional (a silent `NaN` or divide-by-zero should stop the run, not hide in logs), but any new numerical code in that module must handle near-singular or zero-denominator cases explicitly, the way the existing clamps and `try/except RuntimeWarning` guards do. See [Detection → fail-loud policy](./Detection.md#fail-loud-policy).
- **Provenance is checked, not enforced at write time.** The guard hard-fails if a feature file's stored checkpoint path carries a *different* run key, but only *warns* if it carries *none* (legacy/pretrained checkpoints). A checkpoint named without a `bsz.._seed..` key passes with a warning. See [Detection → the provenance guard](./Detection.md#the-provenance-guard).
- **`mahalanobis-solve` is redundant in practice.** Bit-identical to `mahalanobis` on these features, it is dropped from the figures and analysis, but kept as a runnable metric (and verified by `verify_maha_solve.py`) so the equivalence can be checked rather than assumed.
- **`sweep.sh` deliberately omits `set -e`.** This is required for the portable job-throttling (`wait -n` degrades silently on older bash). A failure inside the sweep's own logic will not abort it the way it would in `datamonitor.sh`; per-run failures are surfaced through the `logs/<key>.status` files and the final verdict instead.
- **The extract stage rewrites paths via a symlink.** `get_features.py` writes features to a `numpy_files/` sibling of `DATA_DIR`, and the driver symlinks that to `NUMPY_FILES_DIR`. If a real (non-symlink) directory already exists at the sibling path, the driver warns and features may land elsewhere. See [Architecture → Stage 3](./Architecture.md#stage-3-extract).

For configuration mismatches between the script defaults and the shipped `.env` (Python version, `DM_SEED`, `BOOTSTRAP_N`, and others), see [Installation → known configuration inconsistencies](./Installation.md#known-configuration-inconsistencies).
