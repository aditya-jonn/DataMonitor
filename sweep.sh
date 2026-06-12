#!/usr/bin/env bash
# =============================================================================
# sweep.sh — run multiple DataMonitor pipelines in parallel, one per
# (seed, batch-size) run key, then rebuild the master results table.
#
# Usage:
#   ./sweep.sh                        # run DEFAULT_CONFIGS below
#   ./sweep.sh 2001:128 2002:128      # explicit SEED:BATCH_SIZE pairs
#
#   Stage toggles pass through to every pipeline, e.g. re-run only eval:
#   RUN_TRAIN=0 RUN_EXTRACT=0 ./sweep.sh 2001:128 2002:128
#
# Setup is deliberately NOT pre-run or disabled here: datamonitor.sh stage 1
# is idempotent and flock-serialized, so concurrent pipelines sort it out
# among themselves (first one does the work, the rest fast-skip).
#
# Prerequisites:
#   * .env uses fallback-style assignments (VAR="${VAR:-default}") so the
#     per-run DM_SEED / BATCH_SIZE below actually take effect
#   * datamonitor.sh stage 1 has the flock + atomic cfg.json edits
#
# Rules enforced here:
#   * one pipeline per unique (seed, batch size) key — duplicates refused
#   * at most MAX_PARALLEL pipelines at once (default 4; override via env)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_CONFIGS=( 2001:128 2002:128 2001:256 2002:256 )
MAX_PARALLEL="${MAX_PARALLEL:-4}"

configs=( "$@" )
(( ${#configs[@]} )) || configs=( "${DEFAULT_CONFIGS[@]}" )

# --- validate + refuse duplicate run keys ------------------------------------
declare -A seen
keys=()
for cfg in "${configs[@]}"; do
    IFS=: read -r seed bsz <<< "$cfg"
    if [[ ! "${seed:-}" =~ ^[0-9]+$ || ! "${bsz:-}" =~ ^[0-9]+$ ]]; then
        echo "[sweep] bad config '$cfg' (expected SEED:BATCH_SIZE, e.g. 2001:128)" >&2
        exit 1
    fi
    key="bsz${bsz}_seed${seed}"
    if [[ -n "${seen[$key]:-}" ]]; then
        echo "[sweep] duplicate run key $key — one pipeline per key." >&2
        exit 1
    fi
    seen[$key]=1
    keys+=( "$key" )
done

mkdir -p logs

# --- launch, throttled to MAX_PARALLEL ---------------------------------------
for cfg in "${configs[@]}"; do
    IFS=: read -r seed bsz <<< "$cfg"
    key="bsz${bsz}_seed${seed}"
    log="logs/${key}.log"
    rm -f "logs/${key}.status"
    echo "[sweep] launching $key   (log: $log)"
    # Gate BEFORE launching: poll the live job count. Portable to bash 4.2
    # (wait -n needs >= 4.3 and silently degrades to launch-everything when
    # missing, since this script intentionally runs without set -e).
    if (( MAX_PARALLEL > 0 )); then
        while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
            sleep 5   # polling granularity is negligible vs pipeline runtime
        done
    fi
    (
        DM_SEED="$seed" BATCH_SIZE="$bsz" ./datamonitor.sh > "$log" 2>&1
        echo $? > "logs/${key}.status"
    ) &
done
wait

# --- per-run verdicts ----------------------------------------------------------
fail=0
for key in "${keys[@]}"; do
    rc="$(cat "logs/${key}.status" 2>/dev/null || echo missing)"
    if [[ "$rc" == "0" ]]; then
        echo "[sweep] OK    $key"
    else
        echo "[sweep] FAIL  $key (rc=$rc) — see logs/${key}.log"
        fail=1
    fi
done

# --- rebuild the master table from the per-run results.csv files --------------
PY=python
if [[ -f .env ]]; then set -a; source ./.env; set +a; fi
[[ -n "${VENV_DIR:-}" && -x "$VENV_DIR/bin/python" ]] && PY="$VENV_DIR/bin/python"
"$PY" merge_results.py

if (( fail )); then
    echo "[sweep] one or more runs failed; the master table contains the successful ones." >&2
    exit 1
fi
echo "[sweep] all runs complete."