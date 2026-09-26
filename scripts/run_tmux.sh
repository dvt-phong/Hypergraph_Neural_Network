#!/usr/bin/env bash
# Start scripts/run_all.sh inside a detached tmux session, so the run keeps
# going after the SSH/remote connection is closed.
#
# Usage (same options as run_all.sh):
#   bash scripts/run_tmux.sh
#   bash scripts/run_tmux.sh --skip-prep
#   SEEDS="1 11" bash scripts/run_tmux.sh --no-hsl
#
# Then:
#   tmux attach -t <session>     watch the run      (detach again: Ctrl-b then d)
#   tmux ls                      list sessions
#   tmux kill-session -t <name>  stop a run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux > /dev/null 2>&1; then
    echo "tmux is not installed (Ubuntu: sudo apt install tmux; conda: conda install -c conda-forge tmux)" >&2
    exit 1
fi

# tmux session names cannot contain "." or ":".
SESSION="${SESSION:-hsl_$(date +%d%m%Y_%H%M)}"
if tmux has-session -t "$SESSION" 2> /dev/null; then
    echo "tmux session '$SESSION' already exists: tmux attach -t $SESSION" >&2
    exit 1
fi

# Quote every argument so it reaches run_all.sh unchanged.
printf -v ARGUMENTS " %q" "$@"
printf -v ENVIRONMENT "SEEDS=%q CONDA_ENV=%q PYTHON=%q" \
    "${SEEDS:-1 11 111 1111 11111}" "${CONDA_ENV-hypergraph_nn}" "${PYTHON:-python}"

# The pane stays open after the run so the final message can still be read.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "$ENVIRONMENT bash scripts/run_all.sh$ARGUMENTS; status=\$?; echo; echo \"Run finished (exit \$status). Press Enter to close.\"; read _"

echo "Started tmux session: $SESSION"
echo "  watch:  tmux attach -t $SESSION   (detach: Ctrl-b then d)"
echo "  stop:   tmux kill-session -t $SESSION"
echo "  output: $ROOT/result/<dd-mm-yyyy_HH-MM>/"
