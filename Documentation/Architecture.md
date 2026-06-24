# Architecture

How the pipeline is structured and run: the naming convention that threads through every output, the four stages and what passes between them, the on-disk artifact tree, and the two bash drivers.

---

## The run-key convention

A run is identified by its **batch size `B`** and **random seed `S`**, and most artifacts live in a per-run folder `bsz<B>_seed<S>`. The helpers `run_name(batch_size, seed, metric=None)` and `make_run_dir(...)` in `utils.py` build the name; the optional `metric` splits artifacts into two groups:

- **Metric-independent** (checkpoints, feature matrices): keyed `bsz<B>_seed<S>`. The same embedding feeds every metric, so features are computed once and reused, and every metric sees identical inputs.
- **Metric-dependent** (per-run detection results, control-chart figures): keyed `bsz<B>_<metric>_seed<S>`, so the four metrics never overwrite each other.

So a checkpoint lives in `model_saves/bsz128_seed42/` (no metric) but a result lives in `results/bsz128_mahalanobis_seed42/` (with metric).

---

## The four stages

Each stage is gated by a `RUN_<STAGE>=1/0` toggle (`if [[ "$RUN_<STAGE>" == "1" ]]`), so any prefix of the pipeline can be skipped:

```
setup ──▶ train ──▶ extract ──▶ eval
```

| Stage | What it does | Key | Produced by |
| --- | --- | --- | --- |
| **setup** | Builds the venv, verifies committed shims, writes `cfg.json`. | none | `datamonitor.sh` |
| **train** | Fits the three feature extractors and saves checkpoints. | `bsz<B>_seed<S>` | `train.py` |
| **extract** | Runs each checkpoint over the data splits and caches the feature matrices. | `bsz<B>_seed<S>` | `get_features.py` |
| **eval** | Scores features with every metric, bootstraps detection statistics into the master table, runs the drift simulation. | `bsz<B>_<metric>_seed<S>` | `ood_detection.py`, `merge_results.py` |

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
```

- **Setup → everything.** `cfg.json`'s `data_dir` and `table_path` are read by every entrypoint.
- **Train → Extract.** Checkpoints under `model_saves/bsz<B>_seed<S>/`. Extract auto-discovers the newest matching checkpoint per method, or uses an explicit `*_CKPT` path.
- **Extract → Eval.** Feature matrices and materialised splits under `numpy_files/bsz<B>_seed<S>/`. Each `<method>_features.npz` also stores the checkpoint path that produced it, for a provenance check.
- **Eval → Merge.** Per-run `results.csv` files under `results/bsz<B>_<metric>_seed<S>/`; `merge_results.py` globs them into the single master `ood_bootstrap.csv`.

---

## The artifact tree

A completed run (or sweep) produces this tree (generated directories are git-ignored):

```
DataMonitor/
├── cfg.json                                    # {data_dir, table_path} (written by setup)
├── model_saves/
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT
│       ├── conv-autoencoder_*.pt
│       ├── supervised-cnn_*.pt   (or resnet18_*.pt)
│       └── supervised-ctr_*.pt   (or SupCon_resnet18_*.pt)
├── numpy_files/
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT
│       ├── data_splits.npz                      # Xtr/ytr, Xvl/yvl, Xtt/ytt
│       ├── autoencoder_features.npz             # Ftr, Ftt, + checkpoint path (provenance)
│       ├── cnn_features.npz
│       └── ctr_features.npz
├── results/
│   ├── bsz<B>_<metric>_seed<S>/                 # METRIC-DEPENDENT
│   │   ├── results.csv                          # one row per method
│   │   └── <metric>_<method>_bootstrap.npz      # raw bootstrap arrays
│   └── ood_bootstrap.csv                        # MASTER table (atomically rebuilt)
├── figures/
│   ├── bsz<B>_<metric>_seed<S>/                 # METRIC-DEPENDENT
│   │   └── <metric>_<method>_viz.png            # control chart over the full test set
│   └── bsz<B>_seed<S>/                          # METRIC-INDEPENDENT
│       └── drift_ctr_cosine_figure3.png         # CUSUM drift figure
└── logs/                                        # per-run sweep logs + status files
```

The drift figure always uses the contrastive encoder under cosine scoring, so it is metric-independent; the control charts are metric-dependent.

---

## Experimental grid

The full experiment evaluates every `(method, metric)` pair — 3 × 4 = 12 combinations per run key.

| Feature extractor (method) | OOD embedding | Dim. |
| --- | --- | --- |
| `autoencoder` (conv. AE, unsupervised) | encoder latent vector | 100 |
| `cnn` (ResNet-18, BCE + sigmoid head) | penultimate (pre-fc) vector | 512 |
| `ctr` (SupCon ResNet, contrastive) | L2-normalised encoder output | 512 |

| Metric | Rule |
| --- | --- |
| `cosine` | cosine similarity to centroid (lower ⇒ OOD) |
| `mahalanobis` | Ledoit–Wolf precision matrix (higher ⇒ OOD) |
| `mahalanobis-solve` | Ledoit–Wolf covariance, Cholesky + triangular solve |
| `mahalanobis-pinv` | raw covariance + pseudo-inverse (baseline) |

A binary label `y = 1` marks an in-distribution image and `y = 0` an OOD image throughout the codebase. At the *scoring* step, `ood_detection.py` inverts these (`1 - ytt`), so a `1` there marks OOD; see [Feature Extraction & Detection](./Detection.md).

---

## `datamonitor.sh`: the four-stage driver

`datamonitor.sh` runs the Python entrypoints inside the venv under strict bash (`set -Eeuo pipefail`, plus `IFS=$'\n\t'` — newline + tab, no space — for safe word-splitting on paths). An `ERR` trap updates `CURRENT_STAGE` at the top of each stage and reports which stage a failure died in, so a crash is legible even under `set -e`.

Configuration is resolved once and echoed in a banner before any work: `SCRIPT_DIR` lets the script run from anywhere, `.env` is sourced under `set -a`, and `${VAR:-default}` fills anything `.env` did not set. See [Installation → configuration precedence](./Installation.md#configuration-precedence) for why some `.env` values can be overridden per run and others cannot.

Checkpoint discovery is scoped to the per-run folder `model_saves/bsz<B>_seed<S>/`, so a model from a different batch size or seed is never picked up by accident. Per-method prefix lists (kept in one place, e.g. `supervised-cnn`/`resnet18` both resolving to the CNN) encode the filename styles `train.py` has emitted over the project's life; a `_prefixes_for` helper is shared by the train and extract stages so they agree on naming.

### Stage 1: Setup
Covered in [Installation → the automated setup stage](./Installation.md#the-automated-setup-stage): acquire the `flock` lock, validate the interpreter, build the venv from `requirements.lock.txt`, verify the committed bridge modules and source edits, write `cfg.json` atomically.

### Stage 2: Train
A `train_one` helper runs once per method, skipping a method whose checkpoint exists (unless `FORCE_TRAIN=1`) and otherwise invoking `train.py` with common arguments (dataset, method, learning rate, batch size, seed, max epochs, positive dataset) plus method-specific ones:

```bash
train_one "conv-autoencoder" --c_hid 16 --latent_dim 100
train_one "supervised-cnn"   --base_model "resnet18"
train_one "supervised-ctr"   --base_model "resnet18" --projection "mlp" --temp 0.07
```

### Stage 3: Extract
Resolves each checkpoint path (explicit `*_CKPT`, else newest matching `.pt`), then calls `get_features.py`. It also repairs a path mismatch: `get_features.py` writes features to a `numpy_files/` directory *sibling to* `DATA_DIR`, so if that is not already `NUMPY_FILES_DIR`, the script symlinks them together (comparing canonicalised paths so a relative `DATA_DIR` and an absolute `NUMPY_FILES_DIR` are not falsely flagged).

### Stage 4: Eval
**4a — bootstrap detection.** `ood_detection.py` is invoked once per `(method, metric)` pair, looping over `autoencoder cnn ctr` × `EVAL_METRICS`. `EVAL_METRICS` is split with a locally-scoped `IFS=' '` (the script-wide `IFS` excludes spaces), then `merge_results.py` rebuilds the master CSV; a merge hiccup only *warns*, since a derived view must never fail the eval that produced the real per-run results.

**4b — drift simulation.** An inline Python driver wires together the two `SPC_Charts/` modules, scoring test features by cosine similarity to the contrastive in-distribution centroid. See [Data-Drift Monitoring](./Data-Drift-Monitoring.md).

---

## `sweep.sh`: fan-out across runs

`sweep.sh` launches one full `datamonitor.sh` pipeline per `SEED:BATCH_SIZE` pair, throttled to `MAX_PARALLEL` concurrent jobs, then rebuilds the master table once.

```bash
./sweep.sh                        # run DEFAULT_CONFIGS (2001:128 2002:128 2001:256 2002:256)
./sweep.sh 2001:128 2002:128      # explicit SEED:BATCH_SIZE pairs
RUN_TRAIN=0 RUN_EXTRACT=0 ./sweep.sh 2001:128   # stage toggles pass through
```

It runs under `set -uo pipefail` (deliberately not `-e`). It refuses duplicate run keys up front (two pipelines writing the same `bsz<B>_seed<S>` folder would race), and gates new launches behind a live job count rather than `wait -n` (which needs bash ≥ 4.3 and degrades silently when missing — the reason the script avoids `set -e`). Each pipeline's output goes to `logs/<key>.log` and its exit status to `logs/<key>.status`; after all finish, the sweep prints a per-run `OK`/`FAIL` verdict and runs `merge_results.py` once. If any run failed, the sweep exits non-zero but the master table still contains the successful runs. Setup is neither disabled nor pre-run: stage 1 is idempotent and `flock`-serialised, so concurrent pipelines sort it out among themselves.

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

The [repository layout](../README.md#repository-layout) lists every source file. For conceptual depth: the three extractors, the bridge/registry layer, scoring, and the bootstrap are in [Feature Extraction & Detection](./Detection.md); the drift modules in [Data-Drift Monitoring](./Data-Drift-Monitoring.md); the table and figure builders (`merge_results.py`, `plot_seed_batch.py`) in [Results](./Results.md); and `utils.py` provides `set_seed`, `run_name`, and `make_run_dir`.
