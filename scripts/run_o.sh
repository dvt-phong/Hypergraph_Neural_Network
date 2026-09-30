#!/usr/bin/env bash
# Step O of docs/KE_HOACH_V3.md: the original model (HSL + contrastive, all
# hyperedge families) with the learned W and User hyperedges, screened with
# seed 1 like step B.
#
#   1. U0: add User hyperedges to data/processed/simple/hypergraph.npz if it has
#      none. The kNN lists are reused (not recomputed) and the old bundle is kept
#      as hypergraph_v3.npz. Course/Object/Behavioral hyperedge ids do not change,
#      so every earlier configuration still means the same.
#   2. scripts/run_b.sh with the o* configurations:
#        o1        HSL, Course + Object + Behavioral + User      original model + User
#        o2        o1 + W per family (--family-weights)
#        o3        o2 + W per hyperedge (--edge-weights)          the proposed model
#        o3-skip   o3 + skip connection
#        o3-hgnn   o3 without HSL
#
#   bash scripts/run_o.sh                               all five, seed 1 (about 5-7 hours)
#   ONLY="o3" bash scripts/run_o.sh                     one configuration
#   SEEDS="1 11 111 1111 11111" ONLY="o3" bash scripts/run_o.sh      five seeds
#   RUN_SCRIPT=scripts/run_o.sh bash scripts/run_tmux.sh             the same, inside tmux

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

has_user() {
    "$PYTHON" -c "import sys, numpy as np; sys.exit(0 if 'user_rule' in np.load(sys.argv[1]).files else 1)" "$1"
}

if [[ ! -f "$PROCESSED/hypergraph.npz" ]]; then
    echo "Missing $PROCESSED/hypergraph.npz; run scripts/run_all.sh once without --skip-prep" >&2
    exit 1
fi
if has_user "$PROCESSED/hypergraph.npz"; then
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] U0: hypergraph.npz already has User hyperedges"
else
    if [[ ! -f "$PROCESSED/hypergraph_v3.npz" ]]; then
        cp "$PROCESSED/hypergraph.npz" "$PROCESSED/hypergraph_v3.npz"
    fi
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] U0: adding User hyperedges (old bundle kept as hypergraph_v3.npz)"
    "$PYTHON" -u src/4_hypergraph.py --reuse-neighbors hypergraph_v3.npz || exit 1
fi

export ONLY="${ONLY:-o1 o2 o3 o3-skip o3-hgnn}"
bash scripts/run_b.sh
