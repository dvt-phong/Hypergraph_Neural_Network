#!/usr/bin/env bash
# Step B of docs/IMPROVEMENT_PLAN.md: screen encoder changes for HSL with one
# seed (test std is about 0.001, so one seed is enough to rank them), one
# configuration after another. Each configuration is one scripts/run_all.sh
# call with its own result folder and workbook row. Rerun the best one with
# all seeds afterwards.
#
#   bash scripts/run_b.sh                           every configuration, seed 1
#   ONLY="b12 b12-wd0" bash scripts/run_b.sh        some of them
#   RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh   the same, inside tmux
#
#   name         options (on top of the P0 defaults)
#   b1           --skip-connection
#   b2           --family-weights
#   b12          --skip-connection --family-weights
#   b12-wd5e-5   b12 + --weight-decay 5e-5
#   b12-wd0      b12 + --weight-decay 0
#   b12-do0.2    b12 + --dropout 0.2
#   b12-h256     b12 + --hidden-dim 256
# The name is also the run tag (e.g. hsl_full_skip_fw_b12_seed_1).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS="${SEEDS:-1}"
export TRAIN_SCRIPT=src/8_train.py
ONLY="${ONLY:-b1 b2 b12 b12-wd5e-5 b12-wd0 b12-do0.2 b12-h256}"

FAILED=()
run_config() {
    local name="$1"
    shift
    if [[ " $ONLY " != *" $name "* ]]; then
        return
    fi
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $name: $*"
    if ! NOTE="B, sàng lọc: $name" bash scripts/run_all.sh --skip-prep --tag "$name" "$@"; then
        FAILED+=("$name")
    fi
}

run_config b1         --skip-connection
run_config b2         --family-weights
run_config b12        --skip-connection --family-weights
run_config b12-wd5e-5 --skip-connection --family-weights --weight-decay 5e-5
run_config b12-wd0    --skip-connection --family-weights --weight-decay 0
run_config b12-do0.2  --skip-connection --family-weights --dropout 0.2
run_config b12-h256   --skip-connection --family-weights --hidden-dim 256

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "===== Finished with FAILED configurations: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
echo "===== All configurations finished. Results: result/*/results.csv, docs/ket_qua_thi_nghiem.xlsx"
