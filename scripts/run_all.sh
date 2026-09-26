#!/usr/bin/env bash
# Run the whole pipeline (without downloading) and every seed, then collect results.
#
#   2_preprocess -> 3_features -> 4_hypergraph      once (the split is fixed)
#   8_train --mode both --seeds <seed>              once per seed
#
# Usage (from anywhere):
#   bash scripts/run_all.sh                       full pipeline, all seeds
#   bash scripts/run_all.sh --skip-prep           reuse data/processed, train only
#   bash scripts/run_all.sh --no-hsl              extra options go to 8_train.py
#
# Environment variables:
#   SEEDS="1 11"          seeds to run          (default: 1 11 111 1111 11111)
#   CONDA_ENV=name        conda env to activate (default: hypergraph_nn; "" = none)
#   PYTHON=python3        python executable     (default: python)
#
# Output: result/<dd-mm-yyyy_HH-MM>/
#   pipeline.log          everything, in order
#   logs/<step>.log       one log per step / seed
#   reports/*.json        copies of the train/test reports of this run
#   results.csv           one row per seed + mean + std
#   run_info.txt          command, git commit, GPU

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-1 11 111 1111 11111}"
CONDA_ENV="${CONDA_ENV-hypergraph_nn}"
PYTHON="${PYTHON:-python}"

SKIP_PREP=0
TRAIN_ARGS=()
for argument in "$@"; do
    if [[ "$argument" == "--skip-prep" ]]; then
        SKIP_PREP=1
    else
        TRAIN_ARGS+=("$argument")
    fi
done

# ---------------------------------------------------------------------------
# Result folder: result/dd-mm-yyyy_HH-MM ("/" and ":" are not allowed in names)
# ---------------------------------------------------------------------------
RUN_DIR="result/$(date +%d-%m-%Y_%H-%M)"
if [[ -e "$RUN_DIR" ]]; then
    RUN_DIR="${RUN_DIR}-$(date +%S)"
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/reports"
PIPELINE_LOG="$RUN_DIR/pipeline.log"
MANIFEST="$RUN_DIR/manifest.tsv"
printf "seed\tstatus\tduration_sec\tlog\n" > "$MANIFEST"

say() {
    echo "[$(date '+%d/%m/%Y %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

# Run one command; its output goes to the terminal, its own log, and pipeline.log.
# Returns the command's exit code.
run_step() {
    local name="$1"
    shift
    local step_log="$RUN_DIR/logs/$name.log"
    say "START $name: $*"
    "$@" 2>&1 | tee "$step_log" | tee -a "$PIPELINE_LOG"
    local status="${PIPESTATUS[0]}"
    if [[ "$status" -eq 0 ]]; then
        say "DONE  $name"
    else
        say "FAIL  $name (exit $status), see $step_log"
    fi
    return "$status"
}

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
if [[ -n "$CONDA_ENV" ]] && command -v conda > /dev/null 2>&1; then
    # `conda activate` needs the shell hook in a non-interactive script.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda activate "$CONDA_ENV"; then
        say "Cannot activate conda env '$CONDA_ENV' (set CONDA_ENV=\"\" to skip)"
        exit 1
    fi
fi
if ! "$PYTHON" -c "import numpy, torch, sklearn" > /dev/null 2>&1; then
    say "'$PYTHON' cannot import numpy/torch/sklearn; activate the right environment first"
    exit 1
fi

{
    echo "started:   $(date '+%d/%m/%Y %H:%M:%S')"
    echo "command:   bash scripts/run_all.sh $*"
    echo "seeds:     $SEEDS"
    echo "skip_prep: $SKIP_PREP"
    echo "python:    $(command -v "$PYTHON") ($("$PYTHON" --version 2>&1))"
    echo "conda env: ${CONDA_DEFAULT_ENV:-none}"
    echo "git:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo "gpu:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
    fi
} > "$RUN_DIR/run_info.txt"

say "Run folder: $RUN_DIR"
PIPELINE_STARTED=$SECONDS

# ---------------------------------------------------------------------------
# Steps 2-4 (no download)
# ---------------------------------------------------------------------------
if [[ "$SKIP_PREP" -eq 0 ]]; then
    for raw_file in prediction_data.tar.gz user_info.csv course_info.csv; do
        if [[ ! -f "data/raw/xuetangx/$raw_file" ]]; then
            say "Missing data/raw/xuetangx/$raw_file; run src/1_download.py once first"
            exit 1
        fi
    done
    run_step 2_preprocess "$PYTHON" -u src/2_preprocess.py || exit 1
    run_step 3_features   "$PYTHON" -u src/3_features.py   || exit 1
    run_step 4_hypergraph "$PYTHON" -u src/4_hypergraph.py || exit 1
else
    if [[ ! -f data/processed/simple/hypergraph.npz ]]; then
        say "--skip-prep given but data/processed/simple/hypergraph.npz does not exist"
        exit 1
    fi
    say "Skipping steps 2-4, using existing data/processed/simple"
fi

# ---------------------------------------------------------------------------
# Step 8, one seed at a time. A failed seed does not stop the others.
# ---------------------------------------------------------------------------
FAILED_SEEDS=()
for seed in $SEEDS; do
    seed_started=$SECONDS
    if run_step "train_seed_$seed" "$PYTHON" -u src/8_train.py --mode both --seeds "$seed" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}; then
        status=ok
    else
        status=failed
        FAILED_SEEDS+=("$seed")
    fi
    printf "%s\t%s\t%s\t%s\n" "$seed" "$status" "$((SECONDS - seed_started))" \
        "$RUN_DIR/logs/train_seed_$seed.log" >> "$MANIFEST"
done

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
run_step collect_results "$PYTHON" scripts/collect_results.py "$RUN_DIR"

elapsed=$((SECONDS - PIPELINE_STARTED))
printf -v elapsed_text "%02d:%02d:%02d" $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
echo "finished:  $(date '+%d/%m/%Y %H:%M:%S') (elapsed $elapsed_text)" >> "$RUN_DIR/run_info.txt"

if [[ ${#FAILED_SEEDS[@]} -gt 0 ]]; then
    say "Finished in $elapsed_text with FAILED seeds: ${FAILED_SEEDS[*]}. Results: $RUN_DIR/results.csv"
    exit 1
fi
say "Finished in $elapsed_text. Results: $RUN_DIR/results.csv"
