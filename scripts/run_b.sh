#!/usr/bin/env bash
# Step B of docs/IMPROVEMENT_PLAN.md: screen encoder changes with one seed
# (test std is about 0.001, so one seed is enough to rank them), one
# configuration after another. Each configuration is one scripts/run_all.sh
# call with its own result folder and workbook row. Rerun the best one with
# all seeds afterwards.
#
#   bash scripts/run_b.sh                           default configurations, seed 1
#   ONLY="b12 b12-wd0" bash scripts/run_b.sh        any of the configurations below
#   RUN_SCRIPT=scripts/run_b.sh bash scripts/run_tmux.sh   the same, inside tmux
#
# Step A showed that graph models stop at a temporary plateau (HGNN seeds that
# ran 600 epochs reached AUC 0.846, the ones stopped near epoch 150 only 0.83)
# and that the MLP needs 500-600 epochs. Every configuration here therefore
# trains up to 1000 epochs with patience 60 (300 epochs without improvement).
#
#   name          model  options                                   question
#   b0            HSL    (P0 settings)                             is early stopping the gap?
#   b0-hgnn       HGNN   --no-hsl                                  HSL vs HGNN, same training
#   b1            HSL    --skip-connection                         does skip reach the MLP?
#   b12           HSL    --skip-connection --family-weights
#   b12-hgnn      HGNN   --no-hsl --skip-connection --family-weights
#   -- not run by default --
#   b1-nob        HSL    --skip-connection --families course,object   without Behavioral
#   b1-nob-hgnn   HGNN   --no-hsl --skip-connection --families course,object
#   b1-hgnn       HGNN   --no-hsl --skip-connection                   all families, no W
#   b12-nob-hgnn  HGNN   --no-hsl --skip-connection --family-weights --families course,object
#   fw2-hgnn      HGNN   b12-hgnn again, family weights with their own lr and no decay
#   fw2-nob-hgnn  HGNN   b12-nob-hgnn again, same
#   (runs before 30/09/2026 trained family weights with the main lr and weight decay)
#   b2            HSL    --family-weights
#   b12-wd5e-5    HSL    b12 + --weight-decay 5e-5
#   b12-wd0       HSL    b12 + --weight-decay 0
#   b12-do0.2     HSL    b12 + --dropout 0.2
#   b12-h256      HSL    b12 + --hidden-dim 256
# The name is also the run tag (e.g. hsl_full_skip_fw_b12_seed_1).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS="${SEEDS:-1}"
export TRAIN_SCRIPT=src/8_train.py
ONLY="${ONLY:-b0 b0-hgnn b1 b12 b12-hgnn}"
LONG=(--epochs 1000 --patience 60)

FAILED=()
run_config() {
    local name="$1"
    shift
    if [[ " $ONLY " != *" $name "* ]]; then
        return
    fi
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $name: $*"
    if ! NOTE="B, sàng lọc: $name" bash scripts/run_all.sh --skip-prep --tag "$name" "${LONG[@]}" "$@"; then
        FAILED+=("$name")
    fi
}

run_config b0
run_config b0-hgnn    --no-hsl
run_config b1         --skip-connection
run_config b12        --skip-connection --family-weights
run_config b12-hgnn   --no-hsl --skip-connection --family-weights
run_config b1-nob      --skip-connection --families course,object
run_config b1-nob-hgnn --no-hsl --skip-connection --families course,object
run_config b1-hgnn     --no-hsl --skip-connection
run_config b12-nob-hgnn --no-hsl --skip-connection --family-weights --families course,object
run_config fw2-hgnn     --no-hsl --skip-connection --family-weights
run_config fw2-nob-hgnn --no-hsl --skip-connection --family-weights --families course,object
run_config b2         --family-weights
run_config b12-wd5e-5 --skip-connection --family-weights --weight-decay 5e-5
run_config b12-wd0    --skip-connection --family-weights --weight-decay 0
run_config b12-do0.2  --skip-connection --family-weights --dropout 0.2
run_config b12-h256   --skip-connection --family-weights --hidden-dim 256

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "===== Finished with FAILED configurations: ${FAILED[*]} (see result/*/pipeline.log)"
    exit 1
fi
echo "===== All configurations finished. Results: result/*/results.csv, docs/ket_qua_thi_nghiem.xlsx"
