#!/usr/bin/env bash
# Run src/9_train.py and then src/10_summary.py inside a detached tmux session, so
# training keeps going after the SSH connection is closed. Arguments go to 9_train.py.
#
#   bash scripts/run_tmux.sh --scenario all --seeds 1 11 111 1111 11111
#   bash scripts/run_tmux.sh --scenario all --seeds 1 --epochs 10 --eval-limit 2000
#   PYTHON=/path/to/python bash scripts/run_tmux.sh ...   (default: .venv, else python3)
#
# Then:
#   tmux attach -t <session>     watch the run      (detach again: Ctrl-b then d)
#   tmux kill-session -t <name>  stop the run
# The log is written to outputs/logs/<dd-mm-yyyy_HH-MM>.log.

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

STAMP="$(date +%d-%m-%Y_%H-%M)"
SESSION="${SESSION:-hgnn_${STAMP//-/}}"
LOG="outputs/logs/$STAMP.log"
mkdir -p "$ROOT/outputs/logs"

# Quote every argument so it reaches 9_train.py unchanged.
printf -v ARGUMENTS " %q" "$@"
# The pane stays open after the run so the last lines can still be read.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "{ $PYTHON -u src/9_train.py$ARGUMENTS && $PYTHON src/10_summary.py; } 2>&1 | tee $LOG; echo; echo 'Run finished. Press Enter to close.'; read _"

echo "Started tmux session: $SESSION"
echo "  watch:   tmux attach -t $SESSION   (detach: Ctrl-b then d)"
echo "  stop:    tmux kill-session -t $SESSION"
echo "  log:     $ROOT/$LOG"
echo "  results: $ROOT/outputs/results.csv, summary.csv, ket_qua.xlsx"
