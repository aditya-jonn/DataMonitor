#!/usr/bin/env bash
# =============================================================================
# datamonitor.sh — staged author-reproduction driver for the DataMonitor paper
#
# Runs the ORIGINAL upstream pipeline (train.py / get_features.py /
# ood_detection.py), inside a Python 3.8 venv that matches the paper's
# requirements.txt. Each stage is gated by a RUN_<STAGE>=1/0 env var so you
# can re-enter the pipeline at any step.
#
# Typical workflows
# -----------------
#   # First time, from scratch (~hours on GPU):
#   ./datamonitor.sh
#
#   # I already have the paper's pretrained checkpoints — skip training:
#   RUN_TRAIN=0 \
#   MODEL_SAVES_DIR=/home/kesavan.venkatesh/ai_monitoring/model_saves \
#   DATA_DIR=/home/kesavan.venkatesh/ai_monitoring/data \
#       ./datamonitor.sh
#
#   # Only re-run the bootstrap evaluation:
#   RUN_SETUP=0 RUN_TRAIN=0 RUN_EXTRACT=0 ./datamonitor.sh
#
# Per-stage env vars
# ------------------
#   RUN_SETUP    1=create venv + install upstream requirements + patch broken
#                imports + write cfg.json.  Idempotent.
#   RUN_TRAIN    1=train all three feature extractors (autoencoder, cnn, ctr).
#                Skips any method whose checkpoint is already in MODEL_SAVES_DIR
#                unless FORCE_TRAIN=1.
#   RUN_EXTRACT  1=dump features from the three checkpoints via get_features.py
#                (auto-discovers the newest .pt for each method).
#   RUN_EVAL     1=bootstrap detection (Table 3) + CUSUM drift sim (Figure 3).
#
# Other env vars (all have sensible defaults; override on the command line)
# ------------------------------------------------------------------------
#   REPO_DIR              path to the DataMonitor repo (default: script dir)
#   VENV_DIR              path to the venv (default: $REPO_DIR/.venv-py38)
#   PYTHON_BIN            python interpreter (default: python3.8)
#   VIRTUALENV_PYZ        optional path to virtualenv.pyz (fallback for hosts
#                         where `python3.8 -m venv` is broken)
#   DATA_DIR              MedMNIST .npz location (default: $REPO_DIR/data)
#   MODEL_SAVES_DIR       trained checkpoints  (default: $REPO_DIR/model_saves)
#   RESULTS_DIR           CSV + plot output    (default: $REPO_DIR/results)
#   NUMPY_FILES_DIR       feature .npz cache   (default: $REPO_DIR/numpy_files)
#   FIGURES_DIR           drift plots out      (default: $REPO_DIR/figures)
#   POSITIVE_DATASET      in-distribution view (default: organamnist)
#   EPOCHS                training epochs      (default: 100, paper setting)
#   BATCH_SIZE                                 (default: 128)
#   LEARNING_RATE                              (default: 0.001)
#   BOOTSTRAP_N           ood_detection.py -n  (default: 100)
#   USE_GPUS              CUDA_VISIBLE_DEVICES (default: empty = all)
#   FORCE_TRAIN           1 to retrain even if a checkpoint exists
#   AUTOENCODER_CKPT      explicit ckpt paths; if unset, picked automatically
#   CNN_CKPT              from MODEL_SAVES_DIR by method-prefix + newest mtime.
#   CTR_CKPT
#
# What this script does NOT do
# ----------------------------
#   - It does NOT use the rewritten datamonitor_lib/pipeline.py. This is
#     deliberate — to reproduce paper Table 3 / Figure 3 we run the upstream
#     code with the paper's ResNet-18 architectures and paper's checkpoints.
#   - It does NOT try to use Python > 3.8. The upstream requirements.txt pins
#     numpy 1.19 / torch 1.10 which have no wheels for newer Python.
# =============================================================================
set -Eeuo pipefail
IFS=$'\n\t'

# ── locate ourselves so the script can be invoked from anywhere ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── load .env (if present) BEFORE setting defaults ──────────────────────────
# REPO_DIR must be resolved first so we know where to look. We use SCRIPT_DIR
# as the initial value; if .env itself sets REPO_DIR, that value wins for the
# rest of the script.
#
# `set -a; source .env; set +a` exports every assignment in .env, so .env
# values UNCONDITIONALLY OVERRIDE anything already in the environment. This
# is intentional: .env is the canonical config for this machine, and we don't
# want a stale exported var from the user's shell to silently override it.
# To override .env on a one-off basis, edit .env directly.
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090,SC1091
    source "${REPO_DIR}/.env"
    set +a
fi

# ── defaults (overridable via env or .env) ──────────────────────────────────
RUN_SETUP="${RUN_SETUP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# REPO_DIR was set above; allow .env to have overridden it.
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-py38}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"
# VIRTUALENV_PYZ is optional — only used if present

DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
MODEL_SAVES_DIR="${MODEL_SAVES_DIR:-$REPO_DIR/model_saves}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
NUMPY_FILES_DIR="${NUMPY_FILES_DIR:-$REPO_DIR/numpy_files}"
FIGURES_DIR="${FIGURES_DIR:-$REPO_DIR/figures}"

POSITIVE_DATASET="${POSITIVE_DATASET:-organamnist}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BOOTSTRAP_N="${BOOTSTRAP_N:-100}"
USE_GPUS="${USE_GPUS:-}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

# Random seed (set in .env). Fallback required: set -u aborts on unset vars.
DM_SEED="${DM_SEED:-1001}"
export DM_SEED

# Per-run checkpoint sub-folder that train.py writes to (utils.make_run_dir):
#   $MODEL_SAVES_DIR/bsz<BATCH_SIZE>_seed<SEED>/
RUN_SUBDIR="bsz${BATCH_SIZE}_seed${DM_SEED}"

# Metrics evaluated in stage 4a (space-separated; trim or extend in .env).
EVAL_METRICS="${EVAL_METRICS:-cosine mahalanobis mahalanobis-solve mahalanobis-pinv}"

# Optional explicit checkpoint paths (otherwise auto-discovered)
AUTOENCODER_CKPT="${AUTOENCODER_CKPT:-}"
CNN_CKPT="${CNN_CKPT:-}"
CTR_CKPT="${CTR_CKPT:-}"

# ── small helpers ───────────────────────────────────────────────────────────
banner() {
    local msg="$1"
    local bar="============================================================"
    printf '\n%s\n  %s\n%s\n' "$bar" "$msg" "$bar"
}
say()  { printf "[datamonitor] %s\n" "$*"; }
warn() { printf "[datamonitor] WARNING: %s\n" "$*" >&2; }
die()  {
    printf "[datamonitor] ERROR in stage %s: %s\n" "$CURRENT_STAGE" "$*" >&2
    exit 1
}

check_file() {
    [[ -f "$1" ]] || die "required file missing: $1"
}
check_dir() {
    [[ -d "$1" ]] || die "required directory missing: $1"
}

# Find the newest checkpoint for THIS run. Search only the per-run sub-folder
# train.py writes to ($MODEL_SAVES_DIR/$RUN_SUBDIR), so a model from a different
# batch-size/seed experiment is never picked up by accident.
find_newest_ckpt() {
    [[ -d "$MODEL_SAVES_DIR" ]] || { echo ""; return; }
    local run_dir="$MODEL_SAVES_DIR/$RUN_SUBDIR"
    local prefix hit
    # run-specific folder, newest by mtime, prefixes in priority order
    for prefix in "$@"; do
        hit="$(ls -t "$run_dir"/${prefix}_*.pt 2>/dev/null | head -1 || true)"
        if [[ -n "$hit" ]]; then echo "$hit"; return; fi
    done
    echo ""
}

# Named EXIT trap so failures are legible even with set -e
CURRENT_STAGE="(not started)"
on_error() {
    local rc=$?
    printf "\n[datamonitor] FAILED in stage: %s  (exit %d)\n" "$CURRENT_STAGE" "$rc" >&2
    exit "$rc"
}
trap on_error ERR

# ── echo the resolved config up front, before any work ─────────────────────
banner "Configuration"
cat <<EOF
  RUN_SETUP=$RUN_SETUP  RUN_TRAIN=$RUN_TRAIN  RUN_EXTRACT=$RUN_EXTRACT  RUN_EVAL=$RUN_EVAL
  REPO_DIR        = $REPO_DIR
  VENV_DIR        = $VENV_DIR
  PYTHON_BIN      = $PYTHON_BIN
  DATA_DIR        = $DATA_DIR
  MODEL_SAVES_DIR = $MODEL_SAVES_DIR
  RESULTS_DIR     = $RESULTS_DIR
  NUMPY_FILES_DIR = $NUMPY_FILES_DIR
  FIGURES_DIR     = $FIGURES_DIR
  POSITIVE_DATASET= $POSITIVE_DATASET
  EPOCHS          = $EPOCHS    BATCH_SIZE=$BATCH_SIZE   LR=$LEARNING_RATE
  BOOTSTRAP_N     = $BOOTSTRAP_N
  USE_GPUS        = ${USE_GPUS:-<all>}
  FORCE_TRAIN     = $FORCE_TRAIN
EOF

# Always make sure the output directories exist (cheap, idempotent).
mkdir -p "$DATA_DIR" "$MODEL_SAVES_DIR" "$RESULTS_DIR" "$NUMPY_FILES_DIR" "$FIGURES_DIR"

# =============================================================================
# STAGE 1: SETUP — Create virtual environment and install dependencies
# =============================================================================
CURRENT_STAGE="1/4 setup"

if [[ "$RUN_SETUP" == "1" ]]; then
    banner "Stage 1/4: Environment Setup"

    # Serialize setup across concurrent pipelines: the first process does the
    # work, the rest block here then fast-skip through the idempotent body.
    # The lock auto-releases if the holder crashes.
    exec 9>"$REPO_DIR/.setup.lock"
    flock -x 9

    # Sanity-check the requested Python interpreter exists.
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
        die "$PYTHON_BIN not found on PATH. Install Python 3.11 or set PYTHON_BIN."

    # Confirm it's 3.11.x (the version requirements.lock.txt resolves against).
    pyver="$("$PYTHON_BIN" -c 'import sys;print("%d.%d" % sys.version_info[:2])')"
    if [[ "$pyver" != "3.11" ]]; then
        warn "PYTHON_BIN is Python $pyver, not 3.11 — requirements.lock.txt pins"
        warn "torch 2.6 / numpy 2.x resolved for 3.11; install may fail."
        warn "Set PYTHON_BIN=python3 (3.11) to be safe."
    fi

    # Create venv if missing. Prefer python -m venv; fall back to virtualenv.pyz
    # (useful on systems where ensurepip is unavailable, e.g. some HPC nodes
    # with stripped-down Python installs).
    if [[ ! -d "$VENV_DIR" ]]; then
        if [[ -n "${VIRTUALENV_PYZ:-}" && -f "$VIRTUALENV_PYZ" ]]; then
            say "Creating virtual environment with $VIRTUALENV_PYZ ..."
            "$PYTHON_BIN" "$VIRTUALENV_PYZ" "$VENV_DIR"
        else
            say "Creating virtual environment with $PYTHON_BIN -m venv ..."
            "$PYTHON_BIN" -m venv "$VENV_DIR"
        fi
    else
        say "Virtual environment already exists at $VENV_DIR — skipping creation."
    fi

    # Activate. From this point on, `python` and `pip` refer to the venv.
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    say "Upgrading pip, setuptools, and wheel ..."
    pip install --upgrade pip setuptools wheel --quiet

    # Dependencies are installed from the committed lockfile requirements.lock.txt
    # (exact resolved versions; the --extra-index-url for the +cu113 torch/torchvision
    # wheels is recorded inside that file). Tracked source now, not generated here.
    LOCK_FILE="$REPO_DIR/requirements.lock.txt"
    [[ -f "$LOCK_FILE" ]] || die "missing $LOCK_FILE (the frozen environment is committed source now — check out the full repo)."
    say "Installing requirements from committed lockfile ($LOCK_FILE) ..."
    pip install -r "$LOCK_FILE" --quiet
    pip install --quiet rich   # terminal results table (merge_results.py)

    say "Python: $(python --version)   Pip: $(pip --version | cut -d' ' -f1-2)"

    # ── feature_methods shims are committed source (no longer generated here) ─
    # The six bridge modules (feature_methods/__init__.py, feature_methods/supcon_loss.py,
    # and src/{base,conv_autoencoder,ood_supervised_cnn,ood_supervised_ctr}.py) are now
    # tracked in the repo. Verify they exist instead of regenerating them.
    FM_SRC="$REPO_DIR/feature_methods/src"
    FM_PKG="$REPO_DIR/feature_methods"
    for _f in "$FM_PKG/__init__.py" "$FM_PKG/supcon_loss.py" "$FM_SRC/base.py" "$FM_SRC/conv_autoencoder.py" "$FM_SRC/ood_supervised_cnn.py" "$FM_SRC/ood_supervised_ctr.py"; do
        [[ -f "$_f" ]] || die "missing committed module: $_f (these are tracked source now, not generated — check out the full repo)."
    done
    say "feature_methods modules present (committed source)."

    # ── write cfg.json pointing at our DATA_DIR / RESULTS_DIR ───────────────
    # The upstream train.py, get_features.py, ood_detection.py all read cfg.json
    # for these two keys. We rewrite it from scratch so the user's paths win.
    say "Writing $REPO_DIR/cfg.json ..."
    _cfg_tmp="$REPO_DIR/cfg.json.tmp.$$"   # PID-unique: no shared tmp name to collide on
    cat > "$_cfg_tmp" <<EOF
{
    "data_dir": "$DATA_DIR",
    "table_path": "$RESULTS_DIR/ood_bootstrap.csv"
}
EOF
    mv -f "$_cfg_tmp" "$REPO_DIR/cfg.json"   # atomic rename: readers never see a partial file

    # ── datasets/__init__.py and train.py carry their edits as committed source ─
    # Previously patched here at runtime: augmented two-crop views for supervised-ctr
    # (RandomResizedCrop + RandomRotation) and deterministic seeding in train.py.
    # These are tracked source now, so verify the edits are present (don't apply them).
    grep -q "RandomResizedCrop" "$REPO_DIR/datasets/__init__.py" || die "datasets/__init__.py lacks the contrastive-augmentation edit; expected committed source (re-checkout the repo)."
    grep -q "_seed_everything" "$REPO_DIR/train.py" || die "train.py lacks the deterministic-seeding edit; expected committed source (re-checkout the repo)."
    say "datasets/__init__.py and train.py carry their committed edits."

    exec 9>&-   # close the lock fd -> release the setup lock
    say "Setup complete."
else
    say "[setup] Skipped (RUN_SETUP=0)."
    # Still need to activate the venv for subsequent stages.
    if [[ -d "$VENV_DIR" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        say "Activated existing venv at $VENV_DIR."
    else
        warn "Venv not found at $VENV_DIR and RUN_SETUP=0 — using system Python. Things may break."
    fi
fi

# =============================================================================
# STAGE 2: TRAIN — train autoencoder, supervised-cnn, supervised-ctr
# =============================================================================
CURRENT_STAGE="2/4 train"
 
# Per-method checkpoint filename prefixes, in preference order. Listed here
# in ONE place so adding a new naming scheme means editing one line. The
# patterns reflect what train.py has actually written over time:
#
#   autoencoder_lr*_bsz*_nep*_indist*_time*.pt   (older autoencoder runs)
#   conv-autoencoder_lr*_*.pt                    (current autoencoder)
#   resnet18_lr*_bsz*_nep*_indist*_time*.pt      (supervised-cnn — backbone name only)
#   supervised-cnn_*.pt                          (hypothetical/current)
#   SupCon_resnet18_lr*_decay*_bsz*_temp*_time*.pt  (supervised-ctr — SupCon prefix)
#   supervised-ctr_*.pt                          (hypothetical/current)
AUTOENCODER_PREFIXES=( 'conv-autoencoder' 'autoencoder' )
CNN_PREFIXES=(         'supervised-cnn'   'resnet18' )
CTR_PREFIXES=(         'supervised-ctr'   'SupCon_resnet18' 'SupCon' )
 
# Map method-keyword → prefix-array name (used by train_one / extract).
_prefixes_for() {
    case "$1" in
        autoencoder|conv-autoencoder) printf '%s\n' "${AUTOENCODER_PREFIXES[@]}" ;;
        cnn|supervised-cnn)           printf '%s\n' "${CNN_PREFIXES[@]}" ;;
        ctr|supervised-ctr)           printf '%s\n' "${CTR_PREFIXES[@]}" ;;
        *) die "_prefixes_for: unknown method '$1'" ;;
    esac
}
 
# Helper: train one method, skipping if a checkpoint already exists
# (unless FORCE_TRAIN=1). $1 is the --method value passed to train.py; the
# prefix list to check is looked up automatically from the method name.
train_one() {
    local method="$1"; shift
 
    local existing
    # shellcheck disable=SC2046
    existing="$(find_newest_ckpt $(_prefixes_for "$method"))"
    if [[ -n "$existing" && "$FORCE_TRAIN" != "1" ]]; then
        say "  $method: checkpoint already exists ($(basename "$existing")) — skipping. Set FORCE_TRAIN=1 to retrain."
        return 0
    fi
 
    say "  $method: training ..."
    # Common args. Method-specific args are passed via "$@".
    local common_args=(
        --dataset "MedMNIST-AbdominalCT"
        --method "$method"
        --learning_rate "$LEARNING_RATE"
        --batch_size "$BATCH_SIZE"
        --seed "$DM_SEED"
        --max_epochs "$EPOCHS"
        --positive_dataset "$POSITIVE_DATASET"
    )
    [[ -n "$USE_GPUS" ]] && common_args+=( --use-gpus "$USE_GPUS" )
 
    # Run from REPO_DIR so train.py's relative paths (cfg.json, ./saves/) work.
    ( cd "$REPO_DIR" && python train.py "${common_args[@]}" "$@" )
}
 
if [[ "$RUN_TRAIN" == "1" ]]; then
    banner "Stage 2/4: Train Feature Extractors"
 
    check_file "$REPO_DIR/train.py"
    check_dir  "$DATA_DIR"
 
    # Autoencoder (unsupervised) — matches bash_scripts/autoencoder_runner.sh
    train_one "conv-autoencoder" \
        --c_hid 16 --latent_dim 100
 
    # Supervised CNN — matches bash_scripts/cnn_runner.sh
    train_one "supervised-cnn" \
        --base_model "resnet18"
 
    # Supervised contrastive — matches bash_scripts/ctr_runner.sh
    train_one "supervised-ctr" \
        --base_model "resnet18" --projection "mlp" --temp 0.07
 
    say "Training complete."
else
    say "[train] Skipped (RUN_TRAIN=0)."
fi
 
# =============================================================================
# STAGE 3: EXTRACT — dump features from the three trained checkpoints
# =============================================================================
CURRENT_STAGE="3/4 extract"
 
if [[ "$RUN_EXTRACT" == "1" ]]; then
    banner "Stage 3/4: Extract Features"
 
    check_file "$REPO_DIR/get_features.py"
 
    # Resolve checkpoint paths: explicit env var > newest matching .pt in MODEL_SAVES_DIR.
    # Prefix lists live near the top of the train block — same source of truth.
    # shellcheck disable=SC2046
    [[ -z "$AUTOENCODER_CKPT" ]] && AUTOENCODER_CKPT="$(find_newest_ckpt $(_prefixes_for autoencoder))"
    # shellcheck disable=SC2046
    [[ -z "$CNN_CKPT"         ]] && CNN_CKPT="$(find_newest_ckpt $(_prefixes_for cnn))"
    # shellcheck disable=SC2046
    [[ -z "$CTR_CKPT"         ]] && CTR_CKPT="$(find_newest_ckpt $(_prefixes_for ctr))"
 
    for v in AUTOENCODER_CKPT CNN_CKPT CTR_CKPT; do
        if [[ -z "${!v}" ]]; then
            die "$v is empty and no matching checkpoint found under $MODEL_SAVES_DIR. Run training first or set $v explicitly."
        fi
        check_file "${!v}"
    done
 
    say "  autoencoder ckpt: $(basename "$AUTOENCODER_CKPT")"
    say "  cnn         ckpt: $(basename "$CNN_CKPT")"
    say "  ctr         ckpt: $(basename "$CTR_CKPT")"
 
    # Note: get_features.py writes features to $DATA_DIR/../numpy_files/. Since
    # we already created $NUMPY_FILES_DIR, we make a symlink so its writes land
    # in the right place. (Pure path-rewriting is cleaner than monkey-patching.)
    sibling_numpy="$(dirname "$DATA_DIR")/numpy_files"
    # compare canonical paths: a relative DATA_DIR and an absolute
    # NUMPY_FILES_DIR can be the same directory yet differ as strings,
    # which used to trigger a false 'features may land in the wrong place'
    if [[ "$(realpath -m "$sibling_numpy")" != "$(realpath -m "$NUMPY_FILES_DIR")" ]]; then
        if [[ -e "$sibling_numpy" && ! -L "$sibling_numpy" ]]; then
            warn "$sibling_numpy exists and isn't a symlink — features may land in the wrong place."
        else
            mkdir -p "$NUMPY_FILES_DIR"
            ln -sfn "$NUMPY_FILES_DIR" "$sibling_numpy"
            say "  symlinked $sibling_numpy -> $NUMPY_FILES_DIR"
        fi
    fi
 
    extract_args=(
        --dataset "MedMNIST-AbdominalCT"
        --positive_dataset "$POSITIVE_DATASET"
        --batch_size "$BATCH_SIZE"
        --seed "$DM_SEED"
        --autoencoder_path "$AUTOENCODER_CKPT"
        --cnn_path "$CNN_CKPT"
        --ctr_path "$CTR_CKPT"
    )
    [[ -n "$USE_GPUS" ]] && extract_args+=( --use_gpus "$USE_GPUS" )
    ( cd "$REPO_DIR" && python get_features.py "${extract_args[@]}" )
 
    say "Extract complete. Feature .npz files in $NUMPY_FILES_DIR/$RUN_SUBDIR."
else
    say "[extract] Skipped (RUN_EXTRACT=0)."
fi
 
# =============================================================================
# STAGE 4: EVAL — bootstrap detection (Table 3) + drift simulation (Figure 3)
# =============================================================================
CURRENT_STAGE="4/4 eval"
 
if [[ "$RUN_EVAL" == "1" ]]; then
    banner "Stage 4/4: Evaluation"
 
    check_file "$REPO_DIR/ood_detection.py"
 
    # ── 4a: Bootstrap OOD detection — reproduces paper Table 3 ──────────────
    say "Running bootstrap OOD detection (Table 3) ..."
 
    for method in autoencoder cnn ctr; do
        # NB: script-wide IFS excludes spaces (strict mode, ~line 67), so bare
        # $EVAL_METRICS would NOT split on spaces. Split explicitly; the IFS
        # override is scoped to the read command and leaves global IFS alone.
        IFS=' ' read -r -a _eval_metrics <<< "$EVAL_METRICS"
        for metric in "${_eval_metrics[@]}"; do
            say "  $method × $metric"
            ( cd "$REPO_DIR" && python ood_detection.py \
                --metric "$metric" \
                --method "$method" \
                --batch_size "$BATCH_SIZE" \
                --seed "$DM_SEED" \
                --bootstrap "$BOOTSTRAP_N" )
        done
    done
 
    say ""

    # merge_results.py rebuilds the master CSV and renders the results table.
    # The master is a derived view: a merge hiccup must warn, never fail eval.
    ( cd "$REPO_DIR" && python merge_results.py ) || \
        say "WARNING: results merge failed; per-run results are intact. Rebuild anytime: python merge_results.py"
 
    # ── 4b: Drift simulation — reproduces paper Figure 3 ────────────────────
    # The upstream repo never wires SPC_Charts/data_shift_simulation.py and
    # SPC_Charts/CUSUM_detector.py together — only Examples/*.ipynb does. So
    # we inline a small driver script that uses both modules with the
    # contrastive + cosine features (paper's chosen combo for Figure 3).
    say ""
    say "Running CUSUM drift simulation (Figure 3) ..."
 
    python - <<PY
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
repo = "$REPO_DIR"
numpy_dir = "$NUMPY_FILES_DIR/$RUN_SUBDIR"
figs = "$FIGURES_DIR/$RUN_SUBDIR"
os.makedirs(figs, exist_ok=True)
sys.path.insert(0, repo)
import random
np.random.seed(${DM_SEED:-1001})
random.seed(${DM_SEED:-1001})
from SPC_Charts.data_shift_simulation import simulate_data_shift
from SPC_Charts.CUSUM_detector import CUSUMChangeDetector
 
# Load contrastive features + raw labels for the test set
D = np.load(os.path.join(numpy_dir, "data_splits.npz"))
F = np.load(os.path.join(numpy_dir, "ctr_features.npz"))
Ftr, ytr, Ftt, ytt = F["ctr_Ftr"], D["ytr"], F["ctr_Ftt"], D["ytt"]
 
# In-distribution centroid of training features
Ftr_in = Ftr[ytr == 1]
centroid = Ftr_in.mean(axis=0)
 
# Cosine similarity of every TEST feature to the in-dist centroid
def cos(M, c):
    n = (np.linalg.norm(M, axis=1) + 1e-12) * (np.linalg.norm(c) + 1e-12)
    return np.clip((M @ c) / n, 0.0, 1.0)
test_scores = cos(Ftt, centroid)
 
in_pool  = test_scores[ytt == 1]
out_pool = test_scores[ytt == 0]
print(f"[drift] in-pool n={len(in_pool)}  out-pool n={len(out_pool)}")
 
# Paper Figure 3 setup: 60 days, 100 images/day, shift at day 30,
# pre 0-1% OOD, post 3-5% OOD. We use the midpoint (4%) for post.
daily_avg, _, shift_day, _ = simulate_data_shift(
    in_dist_data=in_pool, out_dist_data=out_pool,
    shift_start_day=31, total_days=60, images_per_day=100,
    shift_percentage=4.0,
)
 
# CUSUM with paper's recommended parameters (k=0.5σ, h≈4σ; see paper Section 3.3)
det = CUSUMChangeDetector(pre_change_days=30, total_days=60)
det.changeDetection(
    CUSUM_data_average_day=daily_avg,
    pre_change_days=30, total_days=60,
    control_limit=4.0,  # h = 4·σ
    k_th=1.0,           # k_th=1 → k = (1·σ)/2 = 0.5σ
    save_plot=False,    # we save to a custom path below
)
fig_path = os.path.join(figs, "drift_ctr_cosine_figure3.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close("all")
print(f"[drift] CUSUM figure saved → {fig_path}")
print(det.summary())
PY
 
    say ""
    say "Eval complete."
    say "  Table 3 CSV: $RESULTS_DIR/ood_bootstrap.csv"
    say "  Figure 3:    $FIGURES_DIR/$RUN_SUBDIR/drift_ctr_cosine_figure3.png"
else
    say "[eval] Skipped (RUN_EVAL=0)."
fi
 
banner "All requested stages finished."