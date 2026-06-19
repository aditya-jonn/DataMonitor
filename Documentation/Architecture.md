# Architecture

How the pipeline is structured and run: the naming convention that threads through every output, the four stages and what passes between them, the on-disk artifact tree, the two bash drivers in depth, and a file-by-file index.

---

## The run-key convention (read this first)

The most important convention is the **run key**, because it names almost every output directory. A run is identified by its **batch size `B`** and **random seed `S`**, and most artifacts live in a per-run sub-folder `bsz<B>_seed<S>`. Two helpers in `utils.py` build the name:

```python
def run_name(batch_size, seed, metric=None):
    """e.g. 'bsz128_mahalanobis_seed42' or 'bsz128_seed42'."""
    parts = [f"bsz{batch_size}"]
    if metric is not None:
        parts.append(str(metric))
    parts.append(f"seed{seed}")
    return "_".join(parts)

def make_run_dir(root, batch_size, seed, metric=None):
    path = os.path.join(root, run_name(batch_size, seed, metric=metric))
    os.makedirs(path, exist_ok=True)
    return path
```

The optional `metric` argument splits artifacts into two groups:

- **Metric-independent artifacts** (checkpoints, feature matrices) are keyed by `bsz<B>_seed<S>` only. The same embedding feeds every metric, so computing features once and reusing them is faster and guarantees every metric sees identical inputs.
- **Metric-dependent artifacts** (per-run detection results, control-chart figures) are keyed by `bsz<B>_<metric>_seed<S>`, so the four metrics never overwrite each other.

That is why a checkpoint lives in `model_saves/bsz128_seed42/` (no metric) but a result lives in `results/bsz128_mahalanobis_seed42/` (with metric).

---

## The four stages

Each stage is gated by an environment toggle, so any prefix of the pipeline can be skipped:

```
setup ──▶ train ──▶ extract ──▶ eval
```

| Stage | What it does | Key | Produced by |
| --- | --- | --- | --- |
| **setup** | Builds the Python environment, verifies committed shims, writes `cfg.json`. | none | `datamonitor.sh` |
| **train** | Fits the three feature extractors and saves checkpoints. | `bsz<B>_seed<S>` | `train.py` |
| **extract** | Runs each checkpoint over the data splits and caches the feature matrices. | `bsz<B>_seed<S>` | `get_features.py` |
| **eval** | Scores those features with every metric, bootstraps detection statistics into the master table, and runs the drift simulation. | `bsz<B>_<metric>_seed<S>` | `ood_detection.py`, `merge_results.py` |

Each stage is wrapped in `if [[ "$RUN_<STAGE>" == "1" ]]`. The common iteration workflow, `RUN_SETUP=0 RUN_TRAIN=0 RUN_EXTRACT=0 ./datamonitor.sh`, re-runs only evaluation against existing features.

---

## Data flow between stages

```
                cfg.json → {data_dir, table_path}
                          │
   ┌──────────┐    ┌───────▼────────┐   ┌──────────────┐    ┌───────────────┐   ┌──────────────┐
   │  Setup   │──▶│     Train      │──▶│   Extract    │──▶│     Eval      │──▶│    Merge     │
   │ venv,cfg │    │  3 extractors  │   │ features/    │    │ 12 (method,   │   │  master      │
   │          │    │                │   │ split        │    │ metric) pairs │   │  table       │
   └──────────┘    └───────┬────────┘   └──────┬───────┘    └──────┬────────┘   └──────┬───────┘
                          │                   │                  │                   │
            model_saves/bsz<B>_seed<S>/   numpy_files/      results/             results/
              <method>_*.pt               bsz<B>_seed<S>/   bsz<B>_<m>_seed<S>/   ood_bootstrap.csv
                                          data_splits.npz   results.csv           (+ CUSUM drift
                                          <method>_         <m>_<method>_          figure)
                                          features.npz      bootstrap.npz
                                                            figures/.../*_viz.png
```

What each arrow carries:

- **Setup → everything.** `cfg.json` carries two keys (`data_dir`, `table_path`) that every Python entrypoint reads.
- **Train → Extract.** Checkpoints (`.pt`) under `model_saves/bsz<B>_seed<S>/`. The extract stage auto-discovers the newest matching checkpoint per method, or uses an explicit `*_CKPT` path.
- **Extract → Eval.** Feature matrices and materialised data splits under `numpy_files/bsz<B>_seed<S>/`. Each `<method>_features.npz` also stores the checkpoint path that produced it, used as a provenance check.
- **Eval → Merge.** Per-run `results.csv` files under `results/bsz<B>_<metric>_seed<S>/`. `merge_results.py` globs them all into the single master `ood_bootstrap.csv`.

---

## The artifact tree

A completed run (or sweep) produces this tree. Generated directories are git-ignored.

```
DataMonitor/
├── cfg.json                                    # {data_dir, table_path}  (written by setup)
│
├── model_saves/
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT key
│       ├── conv-autoencoder_*.pt
│       ├── supervised-cnn_*.pt   (or resnet18_*.pt)
│       └── supervised-ctr_*.pt   (or SupCon_resnet18_*.pt)
│
├── numpy_files/
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT key
│       ├── data_splits.npz                      # Xtr/ytr, Xvl/yvl, Xtt/ytt
│       ├── autoencoder_features.npz             # Ftr, Ftt, + checkpoint path (provenance)
│       ├── cnn_features.npz
│       └── ctr_features.npz
│
├── results/
│   ├── bsz<B>_<metric>_seed<S>/                 # METRIC-DEPENDENT key
│   │   ├── results.csv                          # one row per method
│   │   └── <metric>_<method>_bootstrap.npz      # raw bootstrap arrays
│   └── ood_bootstrap.csv                        # MASTER table (atomically rebuilt)
│
├── figures/
│   ├── bsz<B>_<metric>_seed<S>/                 # METRIC-DEPENDENT key
│   │   └── <metric>_<method>_viz.png            # control chart over the full test set
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT key
│       └── drift_ctr_cosine_figure3.png         # CUSUM drift figure
│
└── logs/                                        # per-run sweep logs + status files
    ├── bsz<B>_seed<S>.log
    └── bsz<B>_seed<S>.status
```

The drift figure always uses the contrastive encoder under cosine scoring, so it is metric-independent and lives under the `bsz<B>_seed<S>` key; the control charts are metric-dependent.

---

## Experimental grid

The full experiment crosses two axes, evaluating every `(method, metric)` pair.

| Feature extractor (method) | Embedding used for OOD | Dim. |
| --- | --- | --- |
| `autoencoder` (conv. AE, unsupervised) | encoder latent vector | 100 |
| `cnn` (ResNet-18, BCE + sigmoid head) | penultimate (pre-fc) vector | 512 |
| `ctr` (SupCon ResNet, contrastive) | L2-normalised encoder output | 512 |

| Metric (scoring function) | Rule |
| --- | --- |
| `cosine` | cosine similarity to centroid (lower ⇒ OOD) |
| `mahalanobis` | Ledoit–Wolf precision matrix (higher ⇒ OOD) |
| `mahalanobis-solve` | Ledoit–Wolf covariance, Cholesky + triangular solve |
| `mahalanobis-pinv` | raw covariance + pseudo-inverse (baseline) |

That gives 3 × 4 = 12 `(method, metric)` combinations per run key. A binary label `y = 1` marks an in-distribution image and `y = 0` an OOD image throughout the codebase. (At the *scoring* step, `ood_detection.py` inverts these via `1 - ytt`, so a `1` there marks an OOD image. See [Feature Extraction & Detection](./Detection.md).)

---

## `datamonitor.sh`: the four-stage driver

`datamonitor.sh` runs the Python entrypoints (`train.py`, `get_features.py`, `ood_detection.py`) inside a controlled venv, under strict bash, with an error trap that reports the stage a failure died in.

### Strict mode and the error trap

```bash
set -Eeuo pipefail
IFS=$'\n\t'
```

`-E` propagates the `ERR` trap into functions and subshells; `-e` exits on any unhandled non-zero return; `-u` aborts on an unset variable (hence the `${VAR:-default}` fallbacks everywhere); `-o pipefail` fails a pipeline if any stage fails. The custom `IFS` (newline + tab, no space) makes word-splitting safe for paths with spaces, with one consequence the eval loop works around (below).

```bash
CURRENT_STAGE="(not started)"
on_error() {
    local rc=$?
    printf "\n[datamonitor] FAILED in stage: %s  (exit %d)\n" "$CURRENT_STAGE" "$rc" >&2
    exit "$rc"
}
trap on_error ERR
```

`CURRENT_STAGE` is updated at the top of each stage (`"1/4 setup"`, `"2/4 train"`, ...), so a crash is legible even under `set -e`.

### Configuration resolution

Configuration is resolved once, up front, and echoed in a banner before any work begins: `SCRIPT_DIR` is computed so the script can run from anywhere, `.env` is sourced under `set -a`, and `${VAR:-default}` fills anything `.env` did not set. See [Installation → the precedence wrinkle](./Installation.md#the-precedence-wrinkle-read-this-before-sweeping) for why some `.env` values can be overridden per run and others cannot.

### Checkpoint discovery

Discovery is scoped to the per-run sub-folder, so a model from a different batch size or seed can never be picked up by accident:

```bash
find_newest_ckpt() {
    [[ -d "$MODEL_SAVES_DIR" ]] || { echo ""; return; }
    local run_dir="$MODEL_SAVES_DIR/$RUN_SUBDIR"   # bsz<B>_seed<S>
    for prefix in "$@"; do                         # prefixes in priority order
        hit="$(ls -t "$run_dir"/${prefix}_*.pt 2>/dev/null | head -1 || true)"
        if [[ -n "$hit" ]]; then echo "$hit"; return; fi
    done
    echo ""
}
```

The per-method prefix lists are kept in one place (so a new naming scheme is a one-line edit) and encode the filename styles `train.py` has emitted over the project's life:

```bash
AUTOENCODER_PREFIXES=( 'conv-autoencoder' 'autoencoder' )
CNN_PREFIXES=(         'supervised-cnn'   'resnet18' )
CTR_PREFIXES=(         'supervised-ctr'   'SupCon_resnet18' 'SupCon' )
```

Both `resnet18_*.pt` and `supervised-cnn_*.pt` resolve to the CNN method. A `_prefixes_for <method>` helper maps a method keyword to its prefix array and is shared by the train and extract stages so they agree on naming.

### Stage 1: Setup

Covered in [Installation → the automated setup stage](./Installation.md#the-automated-setup-stage): acquire the `flock` setup lock, validate the interpreter, build the venv from `requirements.lock.txt`, verify the committed bridge modules and source edits, write `cfg.json` atomically.

### Stage 2: Train

A `train_one` helper runs once per method, skipping a method whose checkpoint exists (unless `FORCE_TRAIN=1`) and otherwise invoking `train.py` with common + method-specific arguments:

```bash
train_one "conv-autoencoder" --c_hid 16 --latent_dim 100
train_one "supervised-cnn"   --base_model "resnet18"
train_one "supervised-ctr"   --base_model "resnet18" --projection "mlp" --temp 0.07
```

The common arguments (dataset, method, learning rate, batch size, seed, max epochs, positive dataset) are assembled once; each invocation runs from `REPO_DIR` so `train.py`'s relative paths resolve. See [Feature Extraction & Detection](./Detection.md) for what each method trains.

### Stage 3: Extract

The extract stage resolves each checkpoint path (explicit `*_CKPT` env var, else newest matching `.pt`), then calls `get_features.py`. It also repairs a path mismatch: `get_features.py` writes features to a `numpy_files/` directory *sibling to* `DATA_DIR`, so if that is not already `NUMPY_FILES_DIR`, the script symlinks them together (comparing canonicalised paths so a relative `DATA_DIR` and an absolute `NUMPY_FILES_DIR` pointing at the same place are not falsely flagged). Pure path-rewriting is cleaner than monkey-patching the Python.

### Stage 4: Eval

**4a. Bootstrap detection (the master table).** `ood_detection.py` is invoked once per `(method, metric)` pair:

```bash
for method in autoencoder cnn ctr; do
    IFS=' ' read -r -a _eval_metrics <<< "$EVAL_METRICS"   # split on spaces LOCALLY
    for metric in "${_eval_metrics[@]}"; do
        python ood_detection.py --metric "$metric" --method "$method" \
            --batch_size "$BATCH_SIZE" --seed "$DM_SEED" --bootstrap "$BOOTSTRAP_N"
    done
done
python merge_results.py
```

The local `IFS=' '` override is needed because the script-wide `IFS` excludes spaces (strict mode); without it the space-separated `EVAL_METRICS` would not split. It is scoped to the `read`, so the global `IFS` is untouched. Afterwards `merge_results.py` rebuilds the master CSV, and a merge hiccup only *warns* (a derived view must never fail the eval that produced the real per-run results).

**4b. Drift simulation (the drift figure).** An inline Python driver (a heredoc) wires together the two `SPC_Charts/` modules that the upstream code never connects in normal execution, scoring test features by cosine similarity to the contrastive in-distribution centroid and feeding the in/out pools to `simulate_data_shift` and `CUSUMChangeDetector`. See [Data-Drift Monitoring](./Data-Drift-Monitoring.md) for the parameters and the maths.

---

## `sweep.sh`: fan-out across runs

`sweep.sh` launches one full `datamonitor.sh` pipeline per `SEED:BATCH_SIZE` pair, throttled to `MAX_PARALLEL` concurrent jobs, then rebuilds the master table once.

```bash
./sweep.sh                        # run DEFAULT_CONFIGS  (2001:128 2002:128 2001:256 2002:256)
./sweep.sh 2001:128 2002:128      # explicit SEED:BATCH_SIZE pairs
RUN_TRAIN=0 RUN_EXTRACT=0 ./sweep.sh 2001:128   # stage toggles pass through to each pipeline
```

It runs under `set -uo pipefail` (deliberately not `-e`, for reasons below).

### Refusing duplicate run keys

Two pipelines writing the same `bsz<B>_seed<S>` folder would race, so the sweep refuses duplicates up front:

```bash
declare -A seen
for cfg in "${configs[@]}"; do
    IFS=: read -r seed bsz <<< "$cfg"
    # ...validate both are integers...
    key="bsz${bsz}_seed${seed}"
    if [[ -n "${seen[$key]:-}" ]]; then
        echo "[sweep] duplicate run key $key — one pipeline per key." >&2
        exit 1
    fi
    seen[$key]=1
done
```

### Throttling

New launches are gated behind a live job count rather than `wait -n`, for portability to older bash (`wait -n` needs bash ≥ 4.3 and silently degrades to launch-everything when missing, which is exactly why the script avoids `set -e`):

```bash
if (( MAX_PARALLEL > 0 )); then
    while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
        sleep 5   # polling granularity is negligible vs pipeline runtime
    done
fi
(
    DM_SEED="$seed" BATCH_SIZE="$bsz" ./datamonitor.sh > "$log" 2>&1
    echo $? > "logs/${key}.status"
) &
```

Each pipeline's output goes to `logs/<key>.log` and its exit status to `logs/<key>.status`. After all pipelines finish, the sweep prints a per-run verdict (`OK`/`FAIL`) from the status files, then runs `merge_results.py` once. If any run failed, the sweep exits non-zero but the master table still contains the successful runs.

### Why setup is not pre-run

Setup is neither disabled nor pre-run by `sweep.sh`. Because stage 1 is idempotent and `flock`-serialised, concurrent pipelines sort it out among themselves: the first does the work, the rest block briefly then fast-skip. This relies on the `flock` lock and the atomic `cfg.json` write.

---

## Common workflows

| Goal | Command |
| --- | --- |
| Full run from scratch | `./datamonitor.sh` |
| Have checkpoints, skip training | `RUN_TRAIN=0 ./datamonitor.sh` |
| Re-run only the bootstrap evaluation | `RUN_SETUP=0 RUN_TRAIN=0 RUN_EXTRACT=0 ./datamonitor.sh` |
| Fast smoke test | set `EPOCHS=5 BOOTSTRAP_N=20 FORCE_TRAIN=1` in `.env`, then `./datamonitor.sh` |
| Sweep seeds × batch sizes | `./sweep.sh 1001:128 1002:128 1001:256 1002:256` |
| Rebuild the master table anytime | `python merge_results.py` |

---

## File reference

A file-by-file index. Follow the cross-links for conceptual depth.

### Orchestration

| File | Role |
| --- | --- |
| `datamonitor.sh` | The four-stage driver, detailed above. |
| `sweep.sh` | Parallel multi-run driver, detailed above. |
| `.env` | Per-machine config, auto-loaded under `set -a`; soft (`${VAR:-default}`) and hard assignments. See [Installation](./Installation.md#configuration-the-env-file). |
| `requirements.lock.txt` / `requirements.txt` | The committed modern lockfile (Python 3.11, torch 2.6) the pipeline installs from, and the original upstream pin set (Python ≤3.10, torch 1.10) for reference. See [Installation](./Installation.md#prerequisites). |

### Python entrypoints

**`train.py` (stage 2).** Parses CLI options into an `options` dict, derives the in-distribution view name from `positive_dataset` (`organamnist → "Axial"`), builds a per-run checkpoint folder, and makes training reproducible before touching data. `_seed_everything(seed)` fixes the Python/numpy/torch RNGs, sets `PYTHONHASHSEED`, seeds CUDA, and forces deterministic cuDNN; `_seed_worker` handles per-worker dataloader seeding; `main()` builds seeded `DataLoader`s then calls `load_model` and `feature_model.train_model(...)`. Default seed is `DM_SEED` or `1001`. See [Feature Extraction & Detection](./Detection.md).

**`get_features.py` (stage 3).** Loads data with `method="none"` (no two-crop transform), materialises the splits with `matrixify` into `data_splits.npz`, then for each of the three checkpoints loads the model in `testing` mode, runs the matching `FeatureSpace`, and saves `<method>_features.npz` with the train/test feature matrices and the originating checkpoint path (for the provenance guard). See [Feature Extraction & Detection](./Detection.md#feature-extraction-stage-3).

**`ood_detection.py` (stage 4a).** The scoring + bootstrap core; promotes `RuntimeWarning` to errors at import. Key functions:

- `compute_cosine_similarity(tr, tt)`: cosine similarity to the centroid.
- `compute_mahalanobis_distance(tr, tt)`: Ledoit–Wolf precision (production), cached + one-time diagnostics.
- `compute_mahalanobis_distance_solve(tr, tt)`: same covariance, Cholesky + triangular solve (no inverse).
- `compute_mahalanobis_distance_pinv(tr, tt)`: raw covariance + pseudo-inverse baseline (cached, float64, clamped).
- `OLD_compute_mahalanobis_distance(tr, tt)`: the original pinv estimator, kept for reference.
- `apply_spc_rules(data, mean, std, metric)`: one-sided Rule 1 (lower tail for cosine, upper tail for Mahalanobis).
- `compute_control_limits(...)`: train scores → mean, std, `UCL = mean+3σ`, `LCL = mean−3σ`.
- `ood_statistics(...)`: the bootstrap (fixed `random.seed(2022)`, 100 points without replacement, confusion-matrix stats).
- `ood_visualization(...)`: the per-run control-chart figure.
- `main()`: loads splits + features, runs the provenance guard, scores, writes the per-run `results.csv`, `*_bootstrap.npz`, and the control chart.

See [Feature Extraction & Detection](./Detection.md).

**`merge_results.py` (stage 4 aggregator).** `load_master()` globs `results/*/results.csv` and sorts; `write_master()` writes atomically (PID-unique temp + `os.replace`); `render()` prints the grouped, best-accuracy-highlighted table (rich or plain); `--quiet` skips rendering. See [Results & Limitations](./Results-and-Limitations.md#reproducing-the-master-table).

**`plot_seed_batch.py`.** Renders the seed × batch-size figure (paper Figure 2) from the master CSV; drops `mahalanobis-solve` (bit-identical duplicate), keeps `mahalanobis-pinv` dashed; flags `--csv`, `--out`, `--metric`, `--print-summary`. See [Results & Limitations](./Results-and-Limitations.md#reproducing-the-seed--batch-size-figure).

**`verify_maha_solve.py`.** Standalone numerical-correctness harness: phase A (always) tests the production solve code on synthetic ill-conditioned data; phase B (if `ood_detection` imports) cross-checks the repo's own functions. See [Feature Extraction & Detection](./Detection.md#verification-harness-verify_maha_solvepy).

**`utils.py`.** `set_seed(random_seed)`, `run_name(batch_size, seed, metric=None)` (the run-key string), and `make_run_dir(...)` (create + return the per-run directory).

### `datasets/`

- **`datasets/__init__.py`.** `load_data(options)` builds train/val/test `AbnominalCTDataset`s; for `method=="supervised-ctr"` it wraps the transform in `TwoCropTransform` (RandomResizedCrop + RandomRotation) for two augmented crops, while `method="none"` yields single images. `matrixify(dset)` flattens a dataset into an `(N, 784)` matrix `X` and a label vector `y`.
- **`datasets/medmnist_abdominalCT.py`.** `AbnominalCTDataset` concatenates `organ{a,c,s}mnist`, shuffles with a local RNG (does not touch global RNG state), and under `label_mode="cheap-supervised"` returns `(image, 1)` for the in-distribution view and `(image, 0)` for the others. Downloads the `.npz` files on first use. (Upstream-derived.)

### `feature_methods/`

- **`__init__.py`, `supcon_loss.py`.** Top-level shims re-exporting `load_model`/`load_eval` and `SupConLoss`/`TwoCropTransform` from `src`.
- **`src/__init__.py`.** The registry: `load_model(options, mode)` and `load_eval(model, ...)` dispatch on the method key. See [the bridge layer](./Detection.md#the-bridge-layer).
- **`src/base.py`, `src/models/base.py`.** `base.py` re-exports `Model`/`FeatureSpace`; `models/base.py` defines the abstract `Model` (optimiser/scheduler scaffold and the abstract `init_model`/`set_loss_function`/`train_one_epoch`/`_key`) and `FeatureSpace`. (Upstream.)
- **`src/conv_autoencoder.py`, `src/Unsuper_conv_autoencoder.py`.** A thin `_key()` wrapper plus the `Encoder`/`Decoder`/`ConvAutoEncoderCore`, the `ConvAutoEncoder` (MSE loss), and `ConvAutoEncoderFeatureSpace` (100-d latent). See [autoencoder](./Detection.md#1-convolutional-autoencoder-conv-autoencoder).
- **`src/ood_supervised_cnn.py`.** `_ResNet18Binary` (ResNet-18 + `Linear(512,2)` + `Sigmoid`, with a `features()` path to the 512-d penultimate vector), `OODSupervisedCNN` (BCE loss), `OODSupervisedCNNFeatureSpace`. See [CNN](./Detection.md#2-ood-supervised-cnn-supervised-cnn).
- **`src/ood_supervised_ctr.py`.** A self-contained CIFAR-style ResNet, the canonical multi-view `_SupConLoss`, `OODSupervisedCTR` (two-crop training), `OODSupervisedCTRFeatureSpace` (L2-normalised 512-d output). See [contrastive](./Detection.md#3-supervised-contrastive-encoder-supervised-ctr).
- **`src/supcon_loss.py`.** The canonical `SupConLoss` (Khosla et al., supports SimCLR fallback) and `TwoCropTransform`.

### `SPC_Charts/`

- **`data_shift_simulation.py`.** `simulate_data_shift(...)` builds a daily time series from in/out score pools with an injected post-shift OOD rate. (Upstream.) See [the drift simulation](./Data-Drift-Monitoring.md#the-drift-simulation-simulate_data_shift).
- **`CUSUM_detector.py`.** `CUSUMChangeDetector` is a high/low-side CUSUM (`k = (k_th·σ)/2`, `h = control_limit·σ`) whose `changeDetection(...)` records false/true positives, detection delay, MTBFA, and false-alarm rate; `plotCUSUM(...)` draws the chart and `summary()` returns the table. (Upstream.) See [the change detector](./Data-Drift-Monitoring.md#the-change-detector-cusumchangedetector).

---

## A note on upstream code

A few modules are invoked by this pipeline but originate upstream, and are documented at the interface level rather than line-by-line: the base classes (`Model`, `FeatureSpace`) in `feature_methods/src/models/base.py`, the data loader and dataset class in `datasets/`, and the two SPC modules in `SPC_Charts/`. Where the flow depends on them, the relevant guide describes their behaviour. See [Feature Extraction & Detection](./Detection.md) and [Data-Drift Monitoring](./Data-Drift-Monitoring.md).
