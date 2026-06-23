# Feature Extraction & OOD Detection

Stage 2 trains three feature extractors, each mapping a medical image to a fixed-length embedding; stage 3 caches those embeddings; stage 4a scores every test image against the in-distribution reference and bootstraps detection statistics into the master table.

Let an input image be `x ∈ ℝ^{3×28×28}` and `f_θ(x) ∈ ℝ^d` the feature it produces. The three methods differ in the map `f_θ`, the training objective, and the dimension `d`. A 28×28 grayscale image is replicated across three channels at the network input for all three.

| Method (`--method`) | Architecture | Objective | OOD embedding | `d` |
| --- | --- | --- | --- | --- |
| `conv-autoencoder` | conv encoder-decoder | MSE reconstruction | encoder latent | 100 |
| `supervised-cnn` | torchvision ResNet-18 + 2-way sigmoid head | BCE (ID vs OOD) | 512-d penultimate (pre-fc) | 512 |
| `supervised-ctr` | CIFAR-style ResNet-18 + MLP projection head | supervised contrastive (SupCon) | 512-d encoder output, L2-normalised | 512 |

---

## The base classes

Both abstract base classes live in `feature_methods/src/models/base.py` (upstream code).

**`Model`** provides the shared training scaffold. Its `__init__` builds the network (`init_model`), sets the loss (`set_loss_function`), creates an `AdamW` optimiser at `learning_rate` and a `MultiStepLR` scheduler that drops the LR 10× at 50% and 75% of `max_epochs`, and tracks `best_loss`/`best_epoch_number`/`current_epoch_number` plus the loss histories. Each subclass supplies four things: `init_model()` (build the network), `set_loss_function()` (return the loss module), `train_one_epoch(train_loader, val_loader)` (one train/validate pass plus best-model bookkeeping), and a static `_key()` (the method's string key, e.g. `"conv-autoencoder"`). The base class provides `train_model` (loops over `max_epochs`), `save_model` (a checkpoint dict with the state dict, full `options`, optimiser state, epoch, and validation loss), and `load`.

**`FeatureSpace`** wraps a trained `Model` and computes embeddings for the train/val/test splits at construction. Each subclass implements `get_features(dset)`: batch the dataset, triple the grayscale channel, run the model's feature path, and return the embedding stack as a NumPy array.

---

## The bridge layer

Several small modules exist only to wire the package together. They are committed source, verified (not generated) at setup.

- **`feature_methods/__init__.py`** re-exports `load_model` (and the rest of `src`) so `from feature_methods import load_model` resolves.
- **`feature_methods/src/__init__.py`** is the **registry**: `load_model(options, mode="training")` dispatches on `options["method"]` (via each class's `_key()`) to build the right `Model` subclass, and loads the checkpoint at `options["save_path"]` when `mode="testing"`; `load_eval(model, train, val, test)` builds the matching `FeatureSpace`.
- **`feature_methods/src/base.py`** re-exports `Model`/`FeatureSpace` from `models/base.py`.
- **`feature_methods/src/conv_autoencoder.py`** subclasses the upstream autoencoder and adds the `_key()` the registry dispatches on (the upstream class lacks it).
- **`feature_methods/supcon_loss.py`** and **`feature_methods/src/supcon_loss.py`** provide the canonical `SupConLoss` and `TwoCropTransform`; the top-level one is a re-export shim so `datasets/__init__.py` can import `TwoCropTransform`.

This indirection lets `train.py` and `get_features.py` stay method-agnostic: they call `load_model`/`load_eval` and the registry picks the implementation.

---

## 1. Convolutional autoencoder (`conv-autoencoder`)

*File: `feature_methods/src/Unsuper_conv_autoencoder.py` (plus the `_key()` wrapper in `conv_autoencoder.py`). Unsupervised.*

An encoder-decoder trained to reconstruct its input. The encoder applies five 3×3 convolutions (three with stride 2, each halving the spatial resolution), grows the channels `3 → c_hid → 2·c_hid`, then flattens and projects to a latent vector. With `c_hid = 16`, a 28×28 input downsamples to a 4×4 map with 32 channels, flattens to 512, and projects to a 100-d latent. The decoder mirrors this with transposed convolutions back to 3×28×28 and a `tanh` output. The objective is pixel-wise mean-squared reconstruction error:

```
z = E(x) ∈ ℝ^100,   x̂ = D(z),   L_AE = (1/N) Σ_i ‖x_i − x̂_i‖²₂
```

- **Best checkpoint:** lowest validation reconstruction loss.
- **OOD embedding:** the 100-d encoder latent, exposed by `features(x) = encoder(x)`.
- **Premise:** a network trained to reconstruct in-distribution images maps them to a compact latent region, whereas poorly-reconstructed off-axis images land elsewhere. (In practice this signal is weak on these views; see [Results & Limitations](./Results-and-Limitations.md).)

---

## 2. OOD-supervised CNN (`supervised-cnn`)

*File: `feature_methods/src/ood_supervised_cnn.py`.*

A torchvision **ResNet-18** (randomly initialised unless `--pretrained`) whose final `fc` is replaced by `Linear(512, 2)` + `Sigmoid`, trained as a binary ID-versus-OOD classifier with binary cross-entropy on one-hot targets:

```
L_BCE = −(1/N) Σ_i Σ_{c∈{0,1}} [ y_{ic} log ŷ_{ic} + (1 − y_{ic}) log(1 − ŷ_{ic}) ]
```

- **Best checkpoint:** selected by validation AUROC (scoring the positive-class output `out[:, 1]`), not loss.
- **OOD embedding:** not the 2-d logit but the 512-d penultimate vector, the output of global average pooling just before the final linear layer (`features(x)` runs the backbone through `avgpool` and flattens).

The 2-dimensional head is too low-rank for the covariance-based Mahalanobis metrics; the penultimate space is full-rank and is where the discriminative structure sits. This matches the forward hook the upstream evaluation code used.

---

## 3. Supervised contrastive encoder (`supervised-ctr`)

*File: `feature_methods/src/ood_supervised_ctr.py`. Self-contained.*

A supervised contrastive encoder on a CIFAR-style ResNet-18: a 3×3 stride-1 stem with no max-pooling, used instead of the torchvision stem whose 7×7 stride-2 convolution and max-pool would over-downsample a 28×28 image. The encoder produces a 512-d embedding; an MLP projection head maps it to a 128-d vector that is L2-normalised onto the unit hypersphere.

Each image is presented as two augmented crops (`RandomResizedCrop(28)` + `RandomRotation(10)`) via a `TwoCropTransform` (wired in `datasets/__init__.py` for `method=="supervised-ctr"`), so a batch of `B` images yields `2B` views. In one step both views are stacked, encoded, projected, regrouped into the `[B, 2, 128]` shape the loss expects, and scored:

```python
images = torch.cat([images[0], images[1]], dim=0)            # [2B, C, H, W]
feats  = self.model(images)                                  # [2B, 128], L2-normalized
f1, f2 = torch.split(feats, [bsz, bsz], dim=0)
feats  = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [B, 2, 128]
loss   = self.loss_function(feats, labels)                   # SupConLoss
```

The supervised contrastive loss, for an anchor view `i` with positive set `P(i)` (other views sharing `i`'s label) and the set `A(i)` of all other views, with temperature `τ = 0.07`:

```
L_SupCon = Σ_i  (−1/|P(i)|) Σ_{p∈P(i)} log [ exp(⟨z_i, z_p⟩/τ) / Σ_{a∈A(i)} exp(⟨z_i, z_a⟩/τ) ]
```

This pulls same-class views together and pushes different-class views apart on the sphere. (For stability the implementation subtracts a per-row maximum from the logits and masks self-comparisons; with neither labels nor a mask it reduces to the unsupervised SimCLR objective.)

- **Best checkpoint:** lowest validation loss.
- **OOD embedding:** the L2-normalised 512-d encoder output (the input to the projection head), not the 128-d projection, which is used only for the loss and discarded after training.

Two departures from the public SupCon repo this descends from: (1) the CIFAR-style ResNet stem above, and (2) the canonical multi-view `SupConLoss` (Khosla et al.), which expects features shaped `[batch, n_views, dim]`, rather than the public repo's flat objective. A reference copy of `SupConLoss`/`TwoCropTransform` lives in `supcon_loss.py`; the extractor also embeds its own identical copy inline so it has no import-time dependency on the bridge package.

> **Batch-size sensitivity.** Because `L_SupCon` depends on having same-class positives within each batch, the contrastive encoder is the extractor whose representation is most sensitive to batch size; the autoencoder and CNN objectives are largely insensitive. The results bear this out: the contrastive encoder collapses at batch size 16. See [Results & Limitations](./Results-and-Limitations.md).

---

## Feature extraction (stage 3)

`get_features.py` orchestrates extraction:

1. Loads the data with `method="none"`, suppressing the two-crop transform so each image is seen once (the contrastive augmentation is for training only).
2. Materialises the three splits as plain matrices via `matrixify` and saves them as `data_splits.npz` (`Xtr/ytr`, `Xvl/yvl`, `Xtt/ytt`), so evaluation reads identical data.
3. For each checkpoint, loads the model in `testing` mode, runs the matching `FeatureSpace` over the splits, and writes the train/test feature matrices together with the checkpoint path, the seed of the downstream provenance check:

```python
cnn_eval = load_eval(cnn_model, train_set, val_set, test_set)
np.savez(os.path.join(numpy_dir, "cnn_features"),
    cnn_Ftr=cnn_eval.train_features,
    cnn_Ftt=cnn_eval.test_features,
    cnn_pth=opt.cnn_path)        # provenance: which checkpoint produced these
```

After this stage the per-run `numpy_files/` folder holds one `data_splits.npz` and three `<method>_features.npz`, and training is no longer needed downstream.

---

## The reference distribution

Given a trained extractor, the in-distribution training features `F_tr = {f_θ(x) : x ∈ ID train}` define a reference distribution with centroid

```
µ = (1/N) Σ_i f_i ,    f_i = f_θ(x_i)
```

`ood_detection.py` restricts the training features to the in-distribution class before scoring (`Ftr_in = Ftr[ytr == 1]`). Each test image then gets a scalar similarity or distance to this reference, and the decision rule decides whether that score is anomalous.

---

## Scoring functions

### Cosine similarity (`cosine`)

```
s_cos(x) = ⟨f_θ(x), µ⟩ / (‖f_θ(x)‖ ‖µ‖) = cos∠(f_θ(x), µ)
```

clipped to `[0, 1]` (a small ε is added for numerical safety). High cosine means alignment with the in-distribution centroid; OOD samples show low similarity. In code, `compute_cosine_similarity` computes the centroid, then `1 − clip(distance.cosine(feature + ε, centroid), 0, 1)` per test point.

### Mahalanobis distance (`mahalanobis`, `mahalanobis-solve`, `mahalanobis-pinv`)

The Mahalanobis distance accounts for the feature covariance `Σ`, measuring displacement in units of standard deviation along each correlated direction:

```
d_M(x) = sqrt( (f_θ(x) − µ)ᵀ Σ⁻¹ (f_θ(x) − µ) )
```

A large `d_M` means the point is far from the in-distribution manifold (OOD). The hard part is `Σ⁻¹`: the empirical feature covariance is severely ill-conditioned (condition number `κ ≈ 10¹⁴`). The pipeline implements three numerically distinct routes to this distance and compares them directly.

---

## The Mahalanobis numerical core

This is the focus of recent work, and the differences are numerical rather than cosmetic. All three variants share `d²(x) = (x − µ)ᵀ Σ⁻¹ (x − µ)` but differ in how they estimate and apply the inverse covariance. Each caches its fit keyed by the identity and a content fingerprint of the training array, so the expensive fit happens once and the bootstrap reuses it. Each prints a one-time diagnostic report on first use.

### Why the raw estimator is fragile

The original code estimated `Σ` by the raw sample covariance and inverted it with a pseudo-inverse (kept as `OLD_compute_mahalanobis_distance`):

```python
centroid = np.mean(tr, axis=0)
inv_cov  = pinv(np.cov(tr, rowvar=False))   # raw covariance, pseudo-inverse
```

For 512-dimensional features from a limited sample, the raw covariance has low effective rank, with tiny or numerically negative eigenvalues. The diagnostics print the condition number and the participation-ratio effective rank:

```
κ = λ_max / λ_min            r_eff = (Σ_i λ_i)² / (Σ_i λ_i²)
```

When eigenvalues are near zero, the pseudo-inverse amplifies any test-point component along those directions by `1/λ`, and that amplified component is squared before entering the distance. The result is unstable distances, and round-off can push the quadratic form slightly negative, which under this module's strict warning policy (below) would crash the whole evaluation.

### (a) `mahalanobis`: Ledoit–Wolf shrinkage (the production metric)

The fix is to shrink the covariance toward a scaled identity, positive definite by construction:

```
Σ_LW = (1 − α)·Ŝ + α·(tr(Ŝ)/d)·I ,    α ∈ [0,1] estimated automatically
```

The additive term floors every eigenvalue, collapsing the condition number and bounding the amplification, while leaving the dominant directions essentially unchanged (`α` is chosen analytically to minimise expected squared error). The production metric fits `LedoitWolf` once (cached), uses its precision matrix, and clamps the quadratic form at zero defensively:

```python
lw = LedoitWolf().fit(tr)                       # alpha = lw.shrinkage_
centroid, precision = lw.location_, lw.get_precision()
distances.append(np.sqrt(max(diff @ precision @ diff.T, 0.0)))
```

On first use it prints a before/after diagnostic: the raw matrix's eigenvalue spread, effective rank, and pinv cutoff/amplification against the shrunk matrix's shrinkage `α` and eigenvalue floor.

### (b) `mahalanobis-solve`: direct solve (no explicit inverse)

`mahalanobis-solve` computes the same distance under the same Ledoit–Wolf covariance but never forms an inverse. It Cholesky-factors `Σ_LW = L Lᵀ` once, then solves a triangular system per test point:

```python
L    = cholesky(lw.covariance_, lower=True)         # Sigma = L L^T
diff = tt - centroid                                # (n_test, d)
y    = solve_triangular(L, diff.T, lower=True)      # solve L y = diff^T
d    = np.sqrt(np.einsum("ij,ij->j", y, y))         # d^2 = ||y||^2
```

The identity that makes this correct:

```
‖y‖² = ‖L⁻¹(x − µ)‖² = (x − µ)ᵀ L⁻ᵀ L⁻¹ (x − µ) = (x − µ)ᵀ Σ_LW⁻¹ (x − µ) = d²(x)
```

Two things follow automatically. First, `d²` is a sum of squares, non-negative by construction (no clamp needed). Second, the triangular solve is vectorised over all test points at once, faster than a per-row precision multiply. The point the comments stress: "solving" cannot rescue a singular system, since a least-squares solve of a singular covariance *is* the pseudo-inverse operation, with the same amplification. The benefit comes from solving the *shrunk, well-conditioned* system, not from solving as such.

### (c) `mahalanobis-pinv`: the baseline

`mahalanobis-pinv` keeps the original raw-covariance, pseudo-inverse estimator as a deliberate comparison baseline, with three controlled deviations so the comparison isolates the estimator rather than incidental differences:

1. the fit is cached (pinv is deterministic, so the numbers are identical with far less work inside the bootstrap);
2. features are promoted to float64 to match the Ledoit–Wolf paths (isolating the estimator, not arithmetic precision);
3. the quadratic form is clamped at zero before the square root. Pinv round-off can go slightly negative, and this module promotes `RuntimeWarning` to a hard error, so without the clamp one bad sample kills the eval. Clamp events are reported once, since they are evidence of the instability the shrinkage paths avoid.

### Correctness relationships

- Routes (a) and (b) use the same `Σ_LW` and must agree to roughly `10⁻⁹` relative error, a correctness check since they differ only in procedure. (On the trained features they come out bit-identical; see [Results & Limitations](./Results-and-Limitations.md).) A larger gap signals a bug.
- Route (c), solving the raw singular system, is mathematically equivalent to the pseudo-inverse, so it is the shrinkage, not the choice of solver, that resolves the conditioning. Evaluating all three side by side exposes the estimator's effect at the level of detection performance.

---

## Control limits and the decision rule

The reference distances are summarised by a Shewhart control chart. From the training scores the pipeline computes the mean `s̄`, the standard deviation `σ`, and three-sigma control limits:

```
UCL = s̄ + 3σ ,    LCL = s̄ − 3σ
```

A test point is flagged as OOD by a one-sided "Rule 1" whose side depends on the metric:

```python
if metric_name == "cosine":
    if data[i] < (mean - 3*std):  violations["Rule 1"].append(i)   # lower tail
elif metric_name.startswith("mahalanobis"):
    if data[i] > (mean + 3*std):  violations["Rule 1"].append(i)   # upper tail
```

Only the tail corresponding to dissimilarity from the reference counts: low cosine similarity is anomalous, high Mahalanobis distance is anomalous. The opposite tail (more in-distribution-typical than average) is not. For the Mahalanobis score, which is non-negative and right-skewed, the lower limit is not meaningful anyway, a further reason the test is one-sided.

---

## The bootstrap (the master-table numbers)

Detection quality is estimated by bootstrapping over the test set. `ood_statistics` selects the metric's scoring function, computes the in-distribution centre and spread once, then resamples the test set `n` times. Each resample draws 100 test points without replacement under a fixed RNG, scores them, applies Rule 1, and tallies a 2×2 confusion matrix:

```python
random.seed(2022)   # fixed bootstrap RNG (Python random, not numpy)
for i in range(n):
    sample = random.sample(range(tt_features.shape[0]), k=100)   # no replacement, k=100
    tt_subset = tt_features[sample, :]
    tt_subset_distances = fxn(tr_features, tt_subset)            # score the subset
    violations = apply_spc_rules(tt_subset_distances, train_mean, train_std, metric)
    # ... build confusion matrix, append accuracy / sensitivity / specificity ...
```

From each confusion matrix:

```
Accuracy    = (tp + tn) / (tp + tn + fp + fn)
Sensitivity = tp / (tp + fn)      # OOD recall
Specificity = tn / (tn + fp)      # ID retention
```

Each quantity is reported as the mean across the `n` draws, together with the band `[mean − std, mean + std]`. That band is one standard deviation (≈ 68% under a normal approximation), not a 95% interval, so the "LCL/UCL" columns in the table are `x̄ ± s`, not percentile confidence bounds. Because the RNG is pinned and the resampling is identical across metrics, the bootstrap is deterministic given the features and metrics are compared on identical resamples (a paired comparison). All variation in detection performance across seeds and batch sizes therefore enters only through the trained extractor `f_θ`; the scoring stage adds no extra stochasticity.

> **Label convention at the scoring step.** Throughout the dataset code `y = 1` is in-distribution and `y = 0` is OOD, but `ood_detection.py` passes `1 - ytt` into `ood_statistics`, inverting the labels so a `1` marks an OOD image for the confusion matrix. The training features are still selected with the original convention (`Ftr[ytr == 1]` = in-distribution).

`specificity` and `sensitivity` are computed inside `try/except RuntimeWarning`: when a confusion-matrix denominator is zero, the numpy divide raises (this module promotes runtime warnings to errors; see below) and the handler records `NaN`, later aggregated with `nanmean`/`nanstd`.

---

## Outputs of stage 4a

Each `ood_detection.py` invocation:

1. Appends or replaces its method's row in the per-run `results/bsz<B>_<metric>_seed<S>/results.csv` (one row per method; re-running a method replaces its stale row).
2. Dumps the raw bootstrap arrays to `<metric>_<method>_bootstrap.npz`.
3. Saves a full-test control-chart figure to `figures/bsz<B>_<metric>_seed<S>/<metric>_<method>_viz.png` via `ood_visualization`, which plots the per-image score, the centre line and control band, marks auto-detected and true OOD points, and prints the accuracy/specificity/sensitivity.

`merge_results.py` then aggregates all per-run `results.csv` into the master `ood_bootstrap.csv`; see [Results & Limitations](./Results-and-Limitations.md).

---

## The provenance guard

A single invocation loads this run's data splits and the requested method's features, then checks that the checkpoint path stored in the feature file carries this run's key:

```python
expected = run_name(options["batch_size"], options["seed"])   # bsz<B>_seed<S>
if expected not in ckpt_pth:
    if re.search(r"bsz\d+_seed\d+", ckpt_pth):
        raise RuntimeError(f"Feature provenance mismatch: ... ({ckpt_pth})")
    warnings.warn("Feature provenance unverified (legacy checkpoint path): ...")
```

A path carrying a different `bsz.._seed..` key is a true mislabelling and hard-fails; a path carrying no key (legacy or paper-pretrained) only warns. This catches the failure mode where features cached for one run are accidentally scored under another run's key.

---

## Fail-loud policy

`ood_detection.py` promotes runtime warnings to errors at import:

```python
warnings.filterwarnings("error", category=RuntimeWarning)
```

This is why the numerical design matters: a silently-`NaN` distance or a divide-by-zero is no longer a buried warning but a hard stop. It also explains the explicit clamps in the Mahalanobis variants and the `try/except RuntimeWarning` guards around the confusion-matrix ratios, the points where benign zero-denominators are turned into `NaN` *on purpose* rather than allowed to crash the run.

---

## Verification harness (`verify_maha_solve.py`)

A standalone harness exercises the production solve code on synthetic features engineered into the same ill-conditioned regime (low effective rank, float32), in two phases.

**A. Standalone (always runs).** Using exact copies of the production solve pieces, on synthetic data, it checks four things without the repo: a **hypothesis probe** (whether a Cholesky of the *raw* covariance even runs, plus a demonstration that a least-squares solve of the raw system equals the pseudo-inverse), **equivalence** (the Ledoit–Wolf solve and precision paths agree to ~`10⁻⁹`), **non-negativity** (the solve path's `d²` is a sum of squares, always ≥ 0), and **timing** (vectorised triangular solve vs the per-row precision loop).

**B. Acceptance (runs only inside the repo venv, when `import ood_detection` succeeds).** Cross-checks the module's own `compute_mahalanobis_distance`, `compute_mahalanobis_distance_solve`, and (if present) `compute_mahalanobis_distance_pinv` against each other and a raw-pinv reference.

Run it from the repo root:

```bash
python verify_maha_solve.py
```
