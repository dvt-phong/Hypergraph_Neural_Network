#!/usr/bin/env bash
# Run the ablation study of the paper on OULAD and then XuetangX inside one detached tmux session:
# M and the 6 ablations, 5 seeds each, then the summary of each dataset.
#
#   M   full model                      A3  w/o User hyperedge
#   H   w/o MLP branch                  A4  w/o self-loop
#   A1  w/o Course hyperedge            W1  w/o adaptive weight (W = I)
#   A2  w/o Object hyperedge
#
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --seeds 1 --epochs 10     (quick test; arguments go to 9_train.py)
#   PYTHON=/path/to/python bash scripts/run_ablation.sh    (default: .venv, else python3)
#
# Output of each dataset in outputs/ABLATION/<dataset>/: results.csv (one row per scenario and
# seed), summary.csv and ket_qua.xlsx (mean ± std), logs/<dd-mm-yyyy_HH-MM>.log.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v tmux > /dev/null 2>&1; then
    echo "tmux is not installed (Ubuntu: sudo apt install tmux)" >&2
    exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

SCENARIOS="M H A1 A2 A3 A4 W1"
STAMP="$(date +%d-%m-%Y_%H-%M)"
SESSION="${SESSION:-ablation_${STAMP//-/}}"

# Quote every argument so it reaches 9_train.py unchanged.
printf -v ARGUMENTS " %q" "$@"

# One command per dataset: train every scenario, summarize, all printed to the log.
COMMANDS=""
for DATASET in oulad xuetangx; do
    if [[ "$DATASET" == "oulad" ]]; then
        CODE="baseline/new_dataset/OULAD"
    else
        CODE="src"
    fi
    RESULTS="outputs/ABLATION/$DATASET"
    LOG="$RESULTS/logs/$STAMP.log"
    mkdir -p "$ROOT/$RESULTS/logs"
    COMMANDS+="{ $PYTHON -u $CODE/9_train.py --scenario $SCENARIOS --results-dir $RESULTS$ARGUMENTS"
    COMMANDS+=" && $PYTHON $CODE/10_summary.py --results-dir $RESULTS; } 2>&1 | tee $LOG; "
done

# The pane stays open after the run so the last lines can still be read.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "${COMMANDS}echo; echo 'Ablation finished. Press Enter to close.'; read _"

echo "Started tmux session: $SESSION"
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b then d)"
echo "  stop:    tmux kill-session -t $SESSION"
echo "  logs:    $ROOT/outputs/ABLATION/{oulad,xuetangx}/logs/$STAMP.log"
echo "  results: $ROOT/outputs/ABLATION/{oulad,xuetangx}/summary.csv, ket_qua.xlsx"
