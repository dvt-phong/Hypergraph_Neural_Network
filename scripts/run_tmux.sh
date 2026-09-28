#!/usr/bin/env bash
# Start scripts/run_all.sh inside a detached tmux session, so the run keeps
# going after the SSH/remote connection is closed.
#
# Usage (same options as run_all.sh; python comes from <project>/.venv, else
# from the activated virtualenv or conda environment, else PYTHON=/path/to/python):
#   bash scripts/run_tmux.sh
#   bash scripts/run_tmux.sh --skip-prep
#   SEEDS="1 11" bash scripts/run_tmux.sh --no-hsl
#   TRAIN_SCRIPT=src/9_baselines.py bash scripts/run_tmux.sh --skip-prep --model gbdt
#   RUN_SCRIPT=scripts/run_p0.sh bash scripts/run_tmux.sh       whole P0 table
#   RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh        step B screening
# Environment passed on: SEEDS, VENV_DIR, PYTHON, TRAIN_SCRIPT, NOTE,
# EXPORT_EXCEL, ONLY, SKIP_PREP (see run_all.sh and run_p0.sh).
#
# Then:
#   tmux attach -t <session>     watch the run      (detach again: Ctrl-b then d)
#   tmux ls                      list sessions
#   tmux kill-session -t <name>  stop a run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux > /dev/null 2>&1; then
    echo "tmux is not installed (Ubuntu: sudo apt install tmux)" >&2
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
# The tmux shell may not have this shell's environment activated, so resolve
# Python here when there is no project .venv: activated virtualenv, then conda.
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    fi
fi
# Pass the settings explicitly: the tmux shell does not inherit this shell's variables.
# Unset variables are passed empty; each script applies its own default
# (e.g. run_all.sh: all seeds, run_b.sh: seed 1).
printf -v ENVIRONMENT "SEEDS=%q VENV_DIR=%q PYTHON=%q TRAIN_SCRIPT=%q NOTE=%q EXPORT_EXCEL=%q ONLY=%q SKIP_PREP=%q" \
    "${SEEDS:-}" "$VENV_DIR" "$PYTHON" "${TRAIN_SCRIPT:-}" \
    "${NOTE:-}" "${EXPORT_EXCEL:-}" "${ONLY:-}" "${SKIP_PREP:-}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_all.sh}"

# The pane stays open after the run so the final message can still be read.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "$ENVIRONMENT bash $RUN_SCRIPT$ARGUMENTS; status=\$?; echo; echo \"Run finished (exit \$status). Press Enter to close.\"; read _"

echo "Started tmux session: $SESSION"
echo "  watch:  tmux attach -t $SESSION   (detach: Ctrl-b then d)"
echo "  stop:   tmux kill-session -t $SESSION"
echo "  output: $ROOT/result/<dd-mm-yyyy_HH-MM>/"
