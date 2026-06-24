# Feature Extraction & OOD Detection

Stage 2 trains three feature extractors, each mapping a medical image to a fixed-length embedding; stage 3 caches those embeddings; stage 4a scores every test image against the in-distribution reference and bootstraps detection statistics into the master table.

Let an input image be `x ∈ ℝ^{3×28×28}` and `f_θ(x) ∈ ℝ^d` the feature it produces. A 28×28 grayscale image is replicated across three channels at the network input for all three methods, which differ in the map `f_θ`, the training objective, and the dimension `d`.

| Method (`--method`) | Architecture | Objective | OOD embedding | `d` |
| --- | --- | --- | --- | --- |
| `conv-autoencoder` | conv encoder-decoder | MSE reconstruction | encoder latent | 100 |
| `supervised-cnn` | torchvision ResNet-18 + 2-way sigmoid head | BCE (ID vs OOD) | 512-d penultimate (pre-fc) | 512 |
| `supervised-ctr` | CIFAR-style ResNet-18 + MLP projection head | supervised contrastive (SupCon) | 512-d encoder output, L2-normalised | 512 |

---

## The base classes

Both abstract base classes live in `feature_methods/src/models/base.py` (upstream code).

**`Model`** provides the shared training scaffold: an `AdamW` optimiser at `learning_rate` and a `MultiStepLR` scheduler that drops the LR 10× at 50% and 75% of `max_epochs`, plus best-loss/epoch bookkeeping. Each subclass supplies `init_model()` (build the network), `set_loss_function()`, `train_one_epoch(train_loader, val_loader)` (one train/validate pass plus best-model bookkeeping), and a static `_key()` (the method's string key). The base provides `train_model` (the epoch loop), `save_model` (a checkpoint dict with the state dict, full `options`, optimiser state, epoch, and validation loss), and `load`.

**`FeatureSpace`** wraps a trained `Model` and computes embeddings for the train/val/test splits at construction. Each subclass implements `get_features(dset)`: batch the dataset, triple the grayscale channel, run the model's feature path, and return the embedding stack as a NumPy array.

---

## The bridge layer

Several small modules wire the package together. They are committed source, verified (not generated) at setup:

- **`feature_methods/__init__.py`** re-exports `load_model` so `from feature_methods import load_model` resolves.
- **`feature_methods/src/__init__.py`** is the **registry**: `load_model(options, mode="training")` dispatches on `options["method"]` (via each class's `_key()`) to build the right `Model` subclass, loading the checkpoint at `options["save_path"]` when `mode="testing"`; `load_eval(model, train, val, test)` builds the matching `FeatureSpace`.
- **`feature_methods/src/base.py`** re-exports `Model`/`FeatureSpace` from `models/base.py`.
- **`feature_methods/src/conv_autoencoder.py`** subclasses the upstream autoencoder and adds the `_key()` the registry dispatches on (the upstream class lacks it).
- **`feature_methods/supcon_loss.py`** re-exports `SupConLoss`/`TwoCropTransform` so `datasets/__init__.py` can import `TwoCropTransform`.

`train.py` and `get_features.py` stay method-agnostic: they call `load_model`/`load_eval` and the registry picks the implementation.

---

## 1. Convolutional autoencoder (`conv-autoencoder`)

*File: `feature_methods/src/Unsuper_conv_autoencoder.py` (plus the `_key()` wrapper in `conv_autoencoder.py`). Unsupervised.*

An encoder-decoder trained to reconstruct its input. The encoder applies five 3×3 convolutions (three stride-2, each halving resolution), growing channels `3 → c_hid → 2·c_hid`, then flattens and projects to a latent. With `c_hid = 16`, a 28×28 input downsamples to a 4×4 × 32 map, flattens to 512, and projects to a 100-d latent; the decoder mirrors this back to 3×28×28 with a `tanh` output. The objective is pixel-wise mean-squared reconstruction error `L_AE = (1/N) Σ ‖x − x̂‖²`.

- **Best checkpoint:** lowest validation reconstruction loss.
- **OOD embedding:** the 100-d encoder latent (`features(x) = encoder(x)`).

---

## 2. OOD-supervised CNN (`supervised-cnn`)

*File: `feature_methods/src/ood_supervised_cnn.py`.*

A torchvision **ResNet-18** (randomly initialised unless `--pretrained`) whose final `fc` is replaced by `Linear(512, 2)` + `Sigmoid`, trained as a binary ID-vs-OOD classifier with binary cross-entropy on one-hot targets.

- **Best checkpoint:** selected by validation AUROC (scoring the positive-class output `out[:, 1]`), not loss.
- **OOD embedding:** not the 2-d logit but the 512-d penultimate vector.

---

## 3. Supervised contrastive encoder (`supervised-ctr`)

*File: `feature_methods/src/ood_supervised_ctr.py`. Self-contained.*

A supervised contrastive encoder on a CIFAR-style ResNet-18: a 3×3 stride-1 stem with no max-pooling. The encoder produces a 512-d embedding; an MLP projection head maps it to a 128-d vector L2-normalised onto the unit hypersphere.

Each image is presented as two augmented crops (`RandomResizedCrop(28)` + `RandomRotation(10)`) via a `TwoCropTransform` (wired in `datasets/__init__.py` for `method=="supervised-ctr"`), so a batch of `B` images yields `2B` views. Both views are encoded and projected, regrouped into the `[B, 2, 128]` shape the loss expects, and scored by the supervised contrastive loss (anchor view `i`, positive set `P(i)` of other views sharing `i`'s label, all-other-views set `A(i)`, temperature `τ = 0.07`):

```
L_SupCon = Σ_i (−1/|P(i)|) Σ_{p∈P(i)} log [ exp(⟨z_i, z_p⟩/τ) / Σ_{a∈A(i)} exp(⟨z_i, z_a⟩/τ) ]
```

This pulls same-class views together and pushes different-class views apart on the sphere.

- **Best checkpoint:** lowest validation loss.
- **OOD embedding:** the L2-normalised 512-d encoder output (the input to the projection head), not the 128-d projection, which is used only for the loss and discarded after training.

---

## Feature extraction (stage 3)

`get_features.py` loads the data with `method="none"` (suppressing the two-crop transform, which is for training only), materialises the three splits as plain matrices via `matrixify` into `data_splits.npz` (`Xtr/ytr`, `Xvl/yvl`, `Xtt/ytt`) so evaluation reads identical data, then for each checkpoint loads the model in `testing` mode, runs the matching `FeatureSpace`, and writes `<method>_features.npz` with the train/test feature matrices and the originating checkpoint path (for the downstream provenance check). After this stage the per-run `numpy_files/` folder holds one `data_splits.npz` and three `<method>_features.npz`, and training is no longer needed.

---

## The reference distribution and scoring

The in-distribution training features define a reference with centroid `µ = (1/N) Σ f_i`. `ood_detection.py` restricts the training features to the in-distribution class first (`Ftr_in = Ftr[ytr == 1]`), then gives each test image a scalar similarity or distance to this reference, which the decision rule judges anomalous or not.

Cosine similarity (`cosine`) is `cos∠(f_θ(x), µ)`, clipped to `[0, 1]` (a small ε for numerical safety). High cosine means alignment with the in-distribution centroid; OOD samples show low similarity.

Mahalanobis distance (`mahalanobis`, `-solve`, `-pinv`) accounts for the feature covariance `Σ`: `d_M(x) = sqrt( (f_θ(x) − µ)ᵀ Σ⁻¹ (f_θ(x) − µ) )`. A large `d_M` means the point is far from the in-distribution manifold. The hard part is `Σ⁻¹`: the empirical feature covariance is severely ill-conditioned (κ ≈ 10¹⁴). The pipeline implements three numerically distinct routes and compares them directly.

---

## The Mahalanobis numerical core

All three variants share `d²(x) = (x − µ)ᵀ Σ⁻¹ (x − µ)` but differ in how they estimate and apply the inverse covariance. Each caches its fit (keyed by the training array's identity and a content fingerprint) so the expensive fit happens once and the bootstrap reuses it, and prints a one-time diagnostic on first use.

For 512-d features from a limited sample, the raw covariance has low effective rank, with tiny or numerically negative eigenvalues. Where eigenvalues are near zero, the pseudo-inverse amplifies any test-point component along those directions by `1/λ`.

### (a) `mahalanobis` — Ledoit–Wolf shrinkage (production)

Shrink the covariance toward a scaled identity, positive definite by construction:

```
Σ_LW = (1 − α)·Ŝ + α·(tr(Ŝ)/d)·I ,    α ∈ [0,1] estimated automatically
```

The additive term floors every eigenvalue, collapsing the condition number and bounding the amplification while leaving the dominant directions essentially unchanged (`α` minimises expected squared error).

### (b) `mahalanobis-solve` — direct solve (no explicit inverse)

Same Ledoit–Wolf covariance, but never forms an inverse: Cholesky-factor `Σ_LW = L Lᵀ` once, then solve a triangular system per test point. With `y = L⁻¹(x − µ)`,

```
‖y‖² = (x − µ)ᵀ L⁻ᵀ L⁻¹ (x − µ) = (x − µ)ᵀ Σ_LW⁻¹ (x − µ) = d²(x)
```

so `d²` is a sum of squares (non-negative, no clamp needed) and the triangular solve vectorises over all test points (faster than a per-row precision multiply). "Solving" does not by itself rescue a singular system: a least-squares solve of a singular covariance *is* the pseudo-inverse, with the same amplification.

### (c) `mahalanobis-pinv` — the baseline

Keeps the raw-covariance pseudo-inverse estimator as a deliberate comparison, with three controlled deviations so the comparison isolates the estimator: the fit is cached; features are promoted to float64 (matching the LW paths); and the quadratic form is clamped at zero before the square root.



Routes (a) and (b) use the same `Σ_LW`. Route (c) solves the raw singular system, so it is mathematically the pseudo-inverse; evaluating all three side by side exposes the estimator's effect on detection performance.

---

## Control limits and the decision rule

The training scores give a mean `s̄`, standard deviation `σ`, and three-sigma control limits `UCL = s̄ + 3σ`, `LCL = s̄ − 3σ`. A test point is flagged OOD by a one-sided "Rule 1" whose side depends on the metric: low cosine similarity is anomalous (`< s̄ − 3σ`), high Mahalanobis distance is anomalous (`> s̄ + 3σ`).

---

## The bootstrap (the master-table numbers)

`ood_statistics` selects the metric's scoring function, computes the in-distribution centre and spread once, then resamples the test set `n` times. Each resample draws 100 test points without replacement under a fixed RNG (`random.seed(2022)`), scores them, applies Rule 1, and tallies a 2×2 confusion matrix, from which:

```
Accuracy = (tp+tn)/(tp+tn+fp+fn)   Sensitivity = tp/(tp+fn)   Specificity = tn/(tn+fp)
```

Each quantity is reported as the mean across the `n` draws with the band `[mean − std, mean + std]` — one standard deviation (≈ 68%), not a 95% interval. Because the RNG is pinned and the resampling is identical across metrics, the bootstrap is deterministic and metrics are compared on identical resamples (a paired comparison); all variation across seeds and batch sizes therefore enters only through the trained extractor `f_θ`.

---

## Outputs of stage 4a

Each `ood_detection.py` invocation appends or replaces its method's row in the per-run `results/bsz<B>_<metric>_seed<S>/results.csv` (one row per method), dumps the raw bootstrap arrays to `<metric>_<method>_bootstrap.npz`, and saves a full-test control-chart figure to `figures/.../<metric>_<method>_viz.png`. `merge_results.py` then aggregates all per-run `results.csv` into the master `ood_bootstrap.csv`; see [Results](./Results.md).