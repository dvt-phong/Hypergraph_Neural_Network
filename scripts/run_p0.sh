#!/usr/bin/env bash
# Run the P0 experiment table of docs/IMPROVEMENT_PLAN.md, one configuration
# after another. Each configuration is one scripts/run_all.sh call over all
# seeds, so it gets its own result/<dd-mm-yyyy_HH-MM>/ folder and its own row
# in docs/ket_qua_thi_nghiem.xlsx. Fast baselines run first.
#
#   bash scripts/run_p0.sh                          every configuration
#   ONLY="gbdt hsl" bash scripts/run_p0.sh          some of them
#   RUN_SCRIPT=scripts/run_p0.sh bash scripts/run_tmux.sh   the same, inside tmux
#
# Data preparation (steps 2-4) runs once, with the first configuration, unless
# SKIP_PREP=1. SEEDS, PYTHON, VENV_DIR are passed on to run_all.sh.
#
#   name     script              configuration                      plan
#   logreg   9_baselines.py      Logistic Regression                E2.1
#   gbdt     9_baselines.py      histogram GBDT                     E2.1
#   mlp      8_train.py          --no-hsl --families self_loop      E2.2
#   hgnn     8_train.py          --no-hsl                           E2.3
#   hsl      8_train.py          HSL                                E1.2 + H3
# Runs of 8_train.py use --tag p0 (e.g. hsl_full_p0_seed_1).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ONLY="${ONLY:-logreg gbdt mlp hgnn hsl}"
PREP_ARGS=()
if [[ "${SKIP_PREP:-0}" == "1" ]]; then
    PREP_ARGS=(--skip-prep)
fi

# Run 1 saved its checkpoints as outputs/runs/hsl_full_seed_*.pt; a plain
# run_all.sh call with default settings would overwrite them. Keep one copy.
if [[ -f outputs/runs/hsl_full_seed_1.pt && ! -d outputs/runs_lan1 ]]; then
    cp -r outputs/runs outputs/runs_lan1
    echo "Copied outputs/runs to outputs/runs_lan1 (checkpoints of run 1)"
fi

FAILED=()
run_config() {
    local name="$1" script="$2" note="$3"
    shift 3
    if [[ " $ONLY " != *" $name "* ]]; then
        return
    fi
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $name: $note"
    if ! TRAIN_SCRIPT="$script" NOTE="$note" \
        bash scripts/run_all.sh ${PREP_ARGS[@]+"${PREP_ARGS[@]}"} "$@"; then
        FAILED+=("$name")
    fi
    PREP_ARGS=(--skip-prep)
}

run_config logreg src/9_baselines.py "E2.1 Logistic Regression" --model logreg
run_config gbdt   src/9_baselines.py "E2.1 GBDT" --model gbdt
run_config mlp    src/8_train.py "E2.2 MLP, cấu hình P0" --no-hsl --families self_loop --tag p0
# Defined before User hyperedges existed: keep the three families of that time.
run_config hgnn   src/8_train.py "E2.3 HGNN, cấu hình P0" --no-hsl --families course,object,behavioral --tag p0
run_config hsl    src/8_train.py "E1.2 + H3: HSL, cấu hình P0" --families course,object,behavioral --tag p0

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "===== Finished with FAILED configurations: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
echo "===== All configurations finished. Results: result/*/results.csv, docs/ket_qua_thi_nghiem.xlsx"
