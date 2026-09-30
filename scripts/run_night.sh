#!/usr/bin/env bash
# Every experiment of docs/KE_HOACH_V3.md, stage 1 and 2, one after another.
# Meant to run unattended inside tmux:
#
#   RUN_SCRIPT=scripts/run_night.sh bash scripts/run_tmux.sh
#   tmux attach -t <session>          watch; Ctrl-b then d to leave it running
#
# Order (most important first, so a run cut short still has the key numbers):
#   0. Graphs: add User hyperedges to hypergraph.npz (old bundle kept as
#      hypergraph_v3.npz), build hypergraph_temporal.npz. kNN is reused.
#   1a. Proposed model, seed 1:  o3, o3-skip, o3-hgnn, o2, o1            (scripts/run_o.sh)
#   1b. Main model + User, seed 1: u1-hgnn, u0-hgnn, u2-hgnn, u1-temporal (scripts/run_b.sh)
#   2.  Seeds 11 111 1111 11111 for o3, o3-skip, u1-hgnn, so they reach five seeds
#       (the summary merges them with the seed-1 run of stage 1)
#   After each stage: scripts/summarize_results.py writes result/tong_hop_ket_qua.xlsx.
#
# A failed configuration does not stop the others; the list of failures is
# printed at the end. Skip a stage with STAGES, e.g. STAGES="1a 1b".
# Rough time on one GPU: HGNN about 40 min per seed, HSL about 1-1.5 h per seed;
# stage 1 about 9-10 h, stage 2 about 14 h.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGES="${STAGES:-0 1a 1b 2}"
PROCESSED=data/processed/simple
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        PYTHON="$VENV_DIR/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi
export PYTHON

say() {
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $*"
}
wants() {
    [[ " $STAGES " == *" $1 "* ]]
}
FAILED=()
summarize() {
    "$PYTHON" scripts/summarize_results.py || say "summarize_results.py failed"
}

say "night run: stages $STAGES, python $PYTHON, git $(git rev-parse --short HEAD 2>/dev/null)"

if wants 0; then
    # run_o.sh adds User to hypergraph.npz when it has none; ONLY=none trains nothing.
    ONLY=none bash scripts/run_o.sh || { say "could not build the User graph; stopping"; exit 1; }
    if [[ ! -f "$PROCESSED/hypergraph_temporal.npz" ]]; then
        say "building hypergraph_temporal.npz"
        "$PYTHON" -u src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz \
            --reuse-neighbors hypergraph_v3.npz || FAILED+=("hypergraph_temporal")
    fi
fi

if wants 1a; then
    say "stage 1a: proposed model, seed 1"
    ONLY="o3 o3-skip o3-hgnn o2 o1" SEEDS=1 bash scripts/run_o.sh || FAILED+=("stage 1a")
    summarize
fi

if wants 1b; then
    say "stage 1b: main model + User, seed 1"
    ONLY="u1-hgnn u0-hgnn u2-hgnn u1-temporal" SEEDS=1 bash scripts/run_b.sh || FAILED+=("stage 1b")
    summarize
fi

if wants 2; then
    say "stage 2: four more seeds for o3, o3-skip, u1-hgnn"
    ONLY="o3 o3-skip u1-hgnn" SEEDS="11 111 1111 11111" bash scripts/run_b.sh || FAILED+=("stage 2")
    summarize
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "finished with failures: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
say "finished. Table: result/tong_hop_ket_qua.xlsx (sheet Kich_ban), result/tong_hop_ket_qua.md"
