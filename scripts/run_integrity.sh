#!/usr/bin/env bash
# Leakage-free main model, and which part carries the prediction: the node's
# own features (MLP) or the hypergraph. Every configuration runs 5 seeds, one
# scripts/run_all.sh call each, then scripts/analyze_contribution.py compares
# each graph model with the MLP on the test split.
#
#   bash scripts/run_integrity.sh                          everything, 5 seeds
#   ONLY="i-mlp i-cou" SEEDS=1 bash scripts/run_integrity.sh
#   RUN_SCRIPT=scripts/run_integrity.sh bash scripts/run_tmux.sh
#
# Integrity: User hyperedges use hypergraph_temporal.npz (only the learner's
# courses that started no later) and --causal (a node only receives from
# hyperedges whose members all started no later), so no information from after
# a node's 35-day window reaches it, in training or evaluation.
#
#   name           model                                         question
#   i-mlp          MLP (self-loops only)                         own features alone
#   i-co           HGNN + skip + W, Course + Object              graph without User
#   i-cou          HGNN + skip + W, Course + Object + User,      leakage-free main model
#                  temporal User, --causal
#   i-cou-noskip   i-cou without the skip branch                 graph branch alone vs MLP
#   i-co-shuffled  i-co on a shuffled hypergraph (same sizes,    is it the neighbors, or
#                  random members)                               just extra capacity?
#
# i-co and i-co-shuffled run without --causal: the shuffled Course hyperedges mix
# courses, so the causal mask would cut them, and on the real Course + Object
# graph (one course per hyperedge) --causal changes nothing.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS="${SEEDS:-1 11 111 1111 11111}"
export TRAIN_SCRIPT=src/8_train.py
ONLY="${ONLY:-i-mlp i-co i-cou i-cou-noskip i-co-shuffled}"
PROCESSED=data/processed/simple
LONG=(--epochs 1000 --patience 60)
GRAPH=(--no-hsl --family-weights)
TEMPORAL=(--families course,object,user --hypergraph hypergraph_temporal.npz --causal)
RUN_LIST=result/integrity_runs.txt

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi
export PYTHON

if [[ " $ONLY " == *" i-cou"* && ! -f "$PROCESSED/hypergraph_temporal.npz" ]]; then
    echo "===== $PROCESSED/hypergraph_temporal.npz is missing; build it first:"
    echo "      $PYTHON -u src/4_hypergraph.py --user-rule temporal --hypergraph-file hypergraph_temporal.npz"
    exit 1
fi

mkdir -p result
FAILED=()
run_config() {
    local name="$1"
    shift
    if [[ " $ONLY " != *" $name "* ]]; then
        return
    fi
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $name: $*"
    if NOTE="$name (seeds $SEEDS)" bash scripts/run_all.sh --skip-prep --tag "$name" "${LONG[@]}" "$@"; then
        # run_all.sh writes to a new result/<date_time> folder; remember which one.
        echo "$name $(ls -td result/*/ | head -n 1)" >> "$RUN_LIST"
    else
        FAILED+=("$name")
    fi
}

run_config i-mlp         --no-hsl --families self_loop
run_config i-co          "${GRAPH[@]}" --skip-connection --families course,object
run_config i-cou         "${GRAPH[@]}" --skip-connection "${TEMPORAL[@]}"
run_config i-cou-noskip  "${GRAPH[@]}" "${TEMPORAL[@]}"
run_config i-co-shuffled "${GRAPH[@]}" --skip-connection --families course,object --shuffle-graph 7

# Latest folder of each configuration in the run list.
folder_of() {
    awk -v name="$1" '$1 == name { folder = $2 } END { print folder }' "$RUN_LIST" 2>/dev/null
}
MLP_DIR="$(folder_of i-mlp)"
if [[ -n "$MLP_DIR" ]]; then
    for name in i-co i-cou i-cou-noskip i-co-shuffled; do
        GRAPH_DIR="$(folder_of "$name")"
        if [[ -n "$GRAPH_DIR" ]]; then
            echo "===== contribution: $name ($GRAPH_DIR) vs i-mlp ($MLP_DIR)"
            "$PYTHON" scripts/analyze_contribution.py "$MLP_DIR" "$GRAPH_DIR" \
                | tee "${GRAPH_DIR%/}/contribution.txt" || FAILED+=("analyze $name")
        fi
    done
fi
"$PYTHON" scripts/summarize_results.py || echo "===== summarize_results.py failed"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "===== Finished with FAILED steps: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
echo "===== All configurations finished. Runs: $RUN_LIST; per-run contribution.txt/.csv"
