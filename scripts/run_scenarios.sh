#!/usr/bin/env bash
# Run the scenarios of scripts/scenarios.py, each as one scripts/run_all.sh call
# (its own result/<date_time> folder, tagged with the scenario code). After
# every scenario run_all.sh rebuilds result/so_thi_nghiem.xlsx.
#
#   bash scripts/run_scenarios.sh                              every scenario, 5 seeds
#   ONLY="M-any M0" bash scripts/run_scenarios.sh              some scenarios
#   ONLY=M0 SEEDS=1 bash scripts/run_scenarios.sh              one seed, to check timing
#   RUN_SCRIPT=scripts/run_scenarios.sh bash scripts/run_tmux.sh    inside tmux
#   python scripts/scenarios.py list                           what each code means
#
# Before training anything:
#   - scenarios.py check: every selected scenario finds its hypergraph bundle
#     with the User rule it needs (hypergraph.npz "any", hypergraph_temporal.npz
#     "temporal"); otherwise nothing runs.
#   - scripts/check_leakage.py on hypergraph_temporal.npz when a selected
#     scenario uses it (about 4 minutes); a failed check stops everything.
#     SKIP_LEAKAGE_CHECK=1 skips it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS="${SEEDS:-1 11 111 1111 11111}"
SKIP_LEAKAGE_CHECK="${SKIP_LEAKAGE_CHECK:-0}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi
export PYTHON

ONLY="${ONLY:-$("$PYTHON" scripts/scenarios.py codes)}"
say() {
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $*"
}

say "scenarios: $ONLY; seeds: $SEEDS"
# shellcheck disable=SC2086
if ! "$PYTHON" scripts/scenarios.py check $ONLY; then
    say "stopped before training: fix the problems above"
    exit 1
fi

if [[ "$SKIP_LEAKAGE_CHECK" != "1" ]]; then
    for code in $ONLY; do
        if [[ "$("$PYTHON" scripts/scenarios.py bundle "$code")" == "hypergraph_temporal.npz" ]]; then
            say "check_leakage.py on hypergraph_temporal.npz (needed by $code)"
            if ! "$PYTHON" -u scripts/check_leakage.py --hypergraph hypergraph_temporal.npz; then
                say "stopped before training: the leakage check failed"
                exit 1
            fi
            break
        fi
    done
fi

FAILED=()
for code in $ONLY; do
    mapfile -t arguments < <("$PYTHON" scripts/scenarios.py args "$code")
    note="$("$PYTHON" scripts/scenarios.py note "$code") (seeds $SEEDS)"
    say "$code: src/6_train.py ${arguments[*]}"
    if ! SCENARIO="$code" NOTE="$note" \
            bash scripts/run_all.sh --skip-prep --tag "$code" "${arguments[@]}"; then
        FAILED+=("$code")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    say "finished with FAILED scenarios: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
say "all scenarios finished; workbook: result/so_thi_nghiem.xlsx"
