# Installation

DataMonitor runs inside a Python virtual environment that the setup stage of `datamonitor.sh` builds, so you normally configure `.env` and let the pipeline create it rather than installing anything by hand. This guide covers the prerequisites, what setup does, every configuration knob, and the known script-vs-`.env` inconsistencies.

---

## Prerequisites

- **Python 3.11.** The committed lockfile (`requirements.lock.txt`) pins `torch 2.6.0+cu124`, `numpy 2.x`, and related wheels resolved for Python 3.11. Setup checks the interpreter version and warns (does not hard-fail) if it is not 3.11.
- **A C toolchain and standard build utilities** for any packages that compile.
- **(Optional) An NVIDIA GPU with CUDA 12.4.** The lockfile installs the `+cu124` CUDA wheels for `torch`/`torchvision` (resolved for an A100, `sm_80`). The pipeline runs on CPU if no GPU is visible; training is just much slower.
- **`flock`** (from `util-linux`), which serialises the setup stage across concurrent pipelines. It is present on essentially all Linux distributions.

> **`requirements.txt` vs `requirements.lock.txt`.** The repo ships both. `requirements.txt` is the *original upstream* pin set (Python ≤3.10, `numpy 1.19`, `torch 1.10`) kept for historical reference; the pipeline does **not** install from it. The environment actually built comes from `requirements.lock.txt`, the modern (Python 3.11) lockfile. Install from the lockfile.

---

## The automated setup stage

Running `./datamonitor.sh` with `RUN_SETUP=1` (the default) performs these steps in order:

1. **Acquire the setup lock.** `exec 9>"$REPO_DIR/.setup.lock"; flock -x 9` serialises setup so that when several pipelines start at once (under `sweep.sh`), the first does the work and the rest block briefly, then fast-skip the idempotent body. The lock releases automatically if the holder crashes.
2. **Validate the interpreter.** Confirms `PYTHON_BIN` exists and is Python 3.11 (warns otherwise).
3. **Create and populate the virtual environment.** Prefers `python -m venv`; falls back to a `virtualenv.pyz` if `VIRTUALENV_PYZ` is set (useful on HPC nodes whose Python lacks `ensurepip`). It upgrades `pip`/`setuptools`/`wheel` and installs from the committed lockfile:
   ```bash
   pip install -r requirements.lock.txt --quiet
   pip install --quiet rich   # used only by merge_results.py for the terminal table
   ```
   Installing from a committed lockfile rather than resolving on the fly means every machine gets byte-identical versions.
4. **Verify the committed bridge modules exist.** Earlier versions generated several "shim" modules at runtime; they are now committed source. Setup *verifies* (does not regenerate) that the six bridge modules are present:
   - `feature_methods/__init__.py`
   - `feature_methods/supcon_loss.py`
   - `feature_methods/src/base.py`
   - `feature_methods/src/conv_autoencoder.py`
   - `feature_methods/src/ood_supervised_cnn.py`
   - `feature_methods/src/ood_supervised_ctr.py`
5. **Verify the committed source edits.** Two edits formerly patched at runtime are now committed and checked by `grep` guards: `datasets/__init__.py` carries the contrastive two-crop augmentation (`grep -q "RandomResizedCrop"`), and `train.py` carries the deterministic-seeding helper (`grep -q "_seed_everything"`).
6. **Write `cfg.json` atomically.** It is written to a PID-unique temp path and then `mv`-renamed into place, so a concurrent reader never sees a half-written file:
   ```json
   {
       "data_dir":   "<DATA_DIR>",
       "table_path": "<RESULTS_DIR>/ood_bootstrap.csv"
   }
   ```
   All three Python entrypoints read these two keys: `data_dir` says where the MedMNIST `.npz` files live, and `table_path` says where `merge_results.py` writes the master table.

Running a later stage with `RUN_SETUP=0` still activates the existing venv (if present) before proceeding.

---

## The dataset

The dataset is the MedMNIST AbdominalCT family: `organamnist` (axial), `organcmnist` (coronal), and `organsmnist` (sagittal). The dataset class downloads the `.npz` files automatically on first use (`download=True`) into `DATA_DIR`, so you need network access on the first run and enough disk for the three files, but no manual fetching.

---

## Configuration: the `.env` file

`datamonitor.sh` auto-loads `.env` from the repository root before applying its own defaults:

```bash
set -a
source "${REPO_DIR}/.env"
set +a
```

`set -a` exports every assignment, so `.env` is the canonical per-machine configuration. The full template is documented inline in `.env`; the most important knobs are below.

### Stage toggles

| Variable | Default | Meaning |
| --- | --- | --- |
| `RUN_SETUP` | `1` | Create venv, install deps, verify shims, write `cfg.json`. |
| `RUN_TRAIN` | `1` | Train all three extractors (slow). Skips a method whose checkpoint exists unless `FORCE_TRAIN=1`. |
| `RUN_EXTRACT` | `1` | Dump features from the three checkpoints. |
| `RUN_EVAL` | `1` | Bootstrap detection table + CUSUM drift figure. |

### Paths

| Variable | Default | Meaning |
| --- | --- | --- |
| `REPO_DIR` | script directory | Where the repo lives. |
| `VENV_DIR` | `$REPO_DIR/.venv-py38`* | Virtual-environment path. |
| `PYTHON_BIN` | `python3.8`* | Interpreter used to create the venv. |
| `DATA_DIR` | `data` | MedMNIST `.npz` location. |
| `MODEL_SAVES_DIR` | `model_saves` | Trained `.pt` checkpoints. |
| `RESULTS_DIR` | `results` | Bootstrap CSV + per-run results. |
| `NUMPY_FILES_DIR` | `numpy_files` | Cached feature `.npz`. |
| `FIGURES_DIR` | `figures` | Control charts + drift figure. |

\* See the **Version note** below. The shipped `.env` overrides these script defaults with the Python-3.11 values (`VENV_DIR=./venv-py311`, `PYTHON_BIN=python3`).

### Training hyperparameters

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPOCHS` | `100` | Training epochs (paper setting). Drop to ~5 for a smoke test. |
| `BATCH_SIZE` | `128` | Batch size (also names the per-run output folders). |
| `LEARNING_RATE` | `0.001` | Optimiser learning rate (fixed across batch sizes; see [Results & Limitations](./Results-and-Limitations.md)). |
| `DM_SEED` | `1001` / `2008`* | Random seed (also names the per-run output folders). |
| `POSITIVE_DATASET` | `organamnist` | Which CT view is in-distribution. |
| `FORCE_TRAIN` | `0` | Retrain even if a checkpoint exists. |
| `DM_NUM_WORKERS` | `0` / `6`* | DataLoader workers per training pipeline. |
| `OMP_NUM_THREADS` | (unset) / `4`* | Cap each pipeline's BLAS threads (matters under `sweep.sh`). |

\* The script's internal default differs from the shipped `.env` value; see the version note.

### Evaluation and hardware

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOOTSTRAP_N` | `100` / `1000`* | Bootstrap resamples for the detection table. |
| `USE_GPUS` | empty (= all) / `0`* | Comma-separated CUDA device IDs → `CUDA_VISIBLE_DEVICES`. |
| `EVAL_METRICS` | `cosine mahalanobis mahalanobis-solve mahalanobis-pinv` | Space-separated metrics to evaluate in stage 4a. |
| `AUTOENCODER_CKPT`, `CNN_CKPT`, `CTR_CKPT` | auto-discovered | Explicit checkpoint paths (otherwise the newest matching `.pt` is used). |

\* See the version note.

---

## The precedence wrinkle (read this before sweeping)

`.env` mixes two assignment styles, which behave differently because the file is sourced under `set -a`:

- **Soft / fallback assignments**, e.g. `DM_SEED="${DM_SEED:-2008}"`. If the variable is already set in the environment, `:-default` expands to the *existing* value, so a value the caller exported survives. This is why `sweep.sh` can set `DM_SEED`/`BATCH_SIZE` per pipeline and have them take effect.
- **Hard assignments**, e.g. `VENV_DIR=./venv-py311`, `PYTHON_BIN=python3`, `DATA_DIR=data`. These always overwrite the environment, so they are fixed per machine.

This reconciles two statements that look contradictory. The driver's header says ".env values unconditionally override the environment", true for the **hard** lines; `sweep.sh` relies on per-run overrides, true for the **soft** lines it touches. Precedence is decided line by line. If you intend to override a setting per run (via `sweep.sh` or `VAR=... ./datamonitor.sh`), make sure it is written in soft `${VAR:-default}` form in `.env`.

---

## Version note: Python 3.8 vs 3.11

An intentional split in the codebase can confuse a first read:

- The **header comment** of `datamonitor.sh` and the script's *internal* defaults (`VENV_DIR=$REPO_DIR/.venv-py38`, `PYTHON_BIN=python3.8`) describe the *original* paper environment (Python 3.8, `torch 1.10`).
- The **active configuration**, the committed `.env` and `requirements.lock.txt`, targets the *modernised* environment (Python 3.11, `torch 2.6`). `.env` sets `VENV_DIR=./venv-py311` and `PYTHON_BIN=python3` as hard assignments, and the setup version check expects 3.11.

The shipped `.env` wins, so a clean checkout builds a **Python 3.11** environment. Treat the 3.8 strings as historical. If you remove or rewrite `.env`, fall back to the lockfile's expectation (3.11), not the script's stale default. Several other defaults also differ between the script and `.env`; see the next section.

---

## Known configuration inconsistencies

Several defaults differ between `datamonitor.sh`'s internal fallbacks and the committed `.env`. As above, the shipped `.env` value is what a clean checkout uses; the script's default applies only if that `.env` line is removed.

| Setting | `datamonitor.sh` default | Shipped `.env` value | Effective on clean checkout |
| --- | --- | --- | --- |
| Python / venv | `python3.8` / `.venv-py38` | `python3` / `./venv-py311` (hard) | **Python 3.11** |
| `DM_SEED` | `1001` | `${DM_SEED:-2008}` (soft) | **2008** (unless caller exports `DM_SEED`) |
| `BOOTSTRAP_N` | `100` | `${BOOTSTRAP_N:-1000}` (soft) | **1000** (unless overridden) |
| `USE_GPUS` | empty (= all GPUs) | `${USE_GPUS:-0}` (soft) | **GPU 0** (unless overridden) |
| `DM_NUM_WORKERS` | `0` (in `train.py`) | `${DM_NUM_WORKERS:-6}` (soft) | **6** (unless overridden) |

Notes:

- The paper used `BOOTSTRAP_N = 100`; the shipped `.env` raises this to 1000 for tighter bootstrap bands. Larger values are slower but do not change the deterministic seed (`random.seed(2022)`), so results stay reproducible.
- The `.env` references a host-specific `VIRTUALENV_PYZ` path and some commented conda lines that are environment-specific; adjust or remove them for your machine.

To make the script self-consistent without `.env`, set the relevant variables explicitly on the command line, or reconcile `.env` and the script defaults to a single source of truth.

---

## Verifying the install

After setup completes you can sanity-check the numerical core without training anything:

```bash
# Standalone numerical-correctness harness (synthetic data, no checkpoints needed).
python verify_maha_solve.py
```

The "STANDALONE" phase always runs and exercises the production Mahalanobis-solve code on synthetic ill-conditioned features. The "ACCEPTANCE" phase also runs if `ood_detection` imports cleanly (you are in the venv at the repo root with `cfg.json` present). See [Feature Extraction & Detection → Verification harness](./Detection.md#verification-harness-verify_maha_solvepy).
