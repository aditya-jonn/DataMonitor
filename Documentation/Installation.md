# Installation

DataMonitor runs inside a Python virtual environment that the setup stage of `datamonitor.sh` builds, so you normally configure `.env` and let the pipeline create it rather than installing anything by hand. This guide covers the prerequisites, what setup does, the configuration knobs, and the known script-vs-`.env` inconsistencies.

---

## Prerequisites

- **Python 3.11.** The committed lockfile (`requirements.lock.txt`) pins `torch 2.6.0+cu124`, `numpy 2.x`, and related wheels resolved for 3.11. Setup checks the interpreter and warns (does not hard-fail) if it is not 3.11.
- **(Optional) an NVIDIA GPU with CUDA 12.4.** The lockfile installs the `+cu124` wheels for `torch`/`torchvision`.

`requirements.txt` is the *original upstream* pin set (Python ≤3.10, `numpy 1.19`, `torch 1.10`), kept for reference only; the pipeline does **not** install from it. The environment is built from `requirements.lock.txt`.

---

## The automated setup stage

`./datamonitor.sh` with `RUN_SETUP=1` (the default) runs these steps in order:

1. **Acquire the setup lock.** `flock -x` on `$REPO_DIR/.setup.lock` serialises setup, so when several pipelines start at once (under `sweep.sh`) the first does the work and the rest block briefly then fast-skip the idempotent body. The lock releases automatically if the holder crashes.
2. **Validate the interpreter** (`PYTHON_BIN` exists, is 3.11).
3. **Create and populate the venv.** Prefers `python -m venv`; falls back to a `virtualenv.pyz` if `VIRTUALENV_PYZ` is set (for HPC nodes lacking `ensurepip`). Upgrades `pip`/`setuptools`/`wheel`, then `pip install -r requirements.lock.txt` plus `rich` (used only by `merge_results.py`). Installing from a committed lockfile means every machine gets identical versions.
4. **Verify the committed bridge modules exist.** Earlier versions generated several shim modules at runtime; they are committed source now, and setup *verifies* (does not regenerate) the six bridge modules under `feature_methods/`.
5. **Verify the committed source edits** via `grep` guards: `datasets/__init__.py` carries the contrastive two-crop augmentation (`RandomResizedCrop`), and `train.py` carries the deterministic-seeding helper (`_seed_everything`).
6. **Write `cfg.json` atomically** (PID-unique temp path, then `mv`), so a concurrent reader never sees a half-written file:
   ```json
   { "data_dir": "<DATA_DIR>", "table_path": "<RESULTS_DIR>/ood_bootstrap.csv" }
   ```
   All three Python entrypoints read these two keys: where the MedMNIST `.npz` files live, and where `merge_results.py` writes the master table.

Running a later stage with `RUN_SETUP=0` still activates the existing venv before proceeding.

---

## The dataset

The dataset is the MedMNIST AbdominalCT family: `organamnist` (axial), `organcmnist` (coronal), `organsmnist` (sagittal). The dataset class downloads the `.npz` files automatically on first use into `DATA_DIR`, so the first run needs network access and disk for three files, but no manual fetching.

---

## Configuration: the `.env` file

`datamonitor.sh` auto-loads `.env` from the repo root under `set -a` (which exports every assignment) before applying its own defaults. The full template is documented inline in `.env`; the main knobs:

### Stage toggles

`1` = run, `0` = skip. `RUN_SETUP`, `RUN_TRAIN`, `RUN_EXTRACT`, and `RUN_EVAL` are all `1` by default; `RUN_TRAIN` skips a method whose checkpoint exists unless `FORCE_TRAIN=1`.

### Paths

| Variable | Default | Meaning |
| --- | --- | --- |
| `REPO_DIR` | script directory | Where the repo lives. |
| `VENV_DIR` | `$REPO_DIR/.venv-py311` | Virtual-environment path. |
| `PYTHON_BIN` | `python3` | Interpreter used to create the venv. |
| `DATA_DIR` | `data` | MedMNIST `.npz` location. |
| `MODEL_SAVES_DIR` | `model_saves` | Trained `.pt` checkpoints. |
| `RESULTS_DIR` | `results` | Bootstrap CSV + per-run results. |
| `NUMPY_FILES_DIR` | `numpy_files` | Cached feature `.npz`. |
| `FIGURES_DIR` | `figures` | Control charts + drift figure. |

### Training hyperparameters

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPOCHS` | `100` | Training epochs (paper setting). Drop to ~5 for a smoke test. |
| `BATCH_SIZE` | `128` | Batch size (also names the per-run output folders). |
| `LEARNING_RATE` | `0.001` | Optimiser learning rate (fixed across batch sizes; see [Results](./Results.md)). |
| `DM_SEED` | `2008` | Random seed (also names the per-run output folders). |
| `POSITIVE_DATASET` | `organamnist` | Which CT view is in-distribution. |
| `FORCE_TRAIN` | `0` | Retrain even if a checkpoint exists. |
| `DM_NUM_WORKERS` | `6` | DataLoader workers per training pipeline. |
| `OMP_NUM_THREADS` | `4` | Cap each pipeline's BLAS threads (matters under `sweep.sh`). |

### Evaluation and hardware

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOOTSTRAP_N` | `100` | Bootstrap resamples for the detection table. |
| `USE_GPUS` | `0` | Comma-separated CUDA device IDs → `CUDA_VISIBLE_DEVICES`. |
| `EVAL_METRICS` | `cosine mahalanobis mahalanobis-solve mahalanobis-pinv` | Metrics to evaluate in stage 4a. |
| `AUTOENCODER_CKPT`, `CNN_CKPT`, `CTR_CKPT` | auto-discovered | Explicit checkpoint paths (otherwise the newest matching `.pt`). |

---

## Configuration precedence

`.env` is sourced under `set -a`, so a soft assignment like `DM_SEED="${DM_SEED:-2008}"` keeps any value the caller already exported, which is how `sweep.sh` and `VAR=... ./datamonitor.sh` override settings per run; a plain assignment like `PYTHON_BIN=python3` always wins and is fixed per machine. Write anything you need to vary per run (notably `DM_SEED` and `BATCH_SIZE`, which `sweep.sh` sets) in the soft `${VAR:-default}` form.