# Run the original CFIN code (github.com/wzfhaha/dropout_prediction, commit a99db2e) on X59, activity
# only (docs/PLAN_X59.md). The changes to the original are in cfin_changes.patch.
#
# Environment (Python 3.7, TensorFlow 1.15; Linux server):
#   conda create -n cfin python=3.7 -y && conda activate cfin
#   pip install -r baseline/CFIN/requirements.txt
# Run from the project root, after src/2_preprocess.py and src/3_features.py (x59_s<seed>.npz):
#   bash baseline/CFIN/run_cfin.sh setup      # clone, reset to the pinned commit, apply the patch (rerun after git pull)
#   bash baseline/CFIN/run_cfin.sh smoke      # seed 1, 1 epoch on the first 4,000 rows of every split
#   bash baseline/CFIN/run_cfin.sh full       # 5 seeds -> outputs/baselines/CFIN/
# OULAD (X55 of baseline/new_dataset/OULAD): DATASET=oulad bash baseline/CFIN/run_cfin.sh smoke|full
#   -> outputs/oulad/baselines/CFIN/
# Windows (local .venv): PY=baseline/CFIN/.venv/Scripts/python.exe bash baseline/CFIN/run_cfin.sh ...
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATASET="${DATASET:-xuetangx}"                       # xuetangx or oulad (docs/PLAN_OULAD_M_BASELINE.md)
if [ "$DATASET" = "oulad" ]; then
    X_DIR="$ROOT/data/processed/oulad"; X_NAME=x55; OUT="$ROOT/outputs/oulad/baselines/CFIN"
else
    X_DIR="$ROOT/data/processed/simple"; X_NAME=x59; OUT="$ROOT/outputs/baselines/CFIN"
fi
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAGE="${1:-full}"

# 1. Original code at the pinned commit, plus our changes. setup always resets the clone to the
#    commit and applies the current patch, so a clone left from an older patch is never used.
if [ "$STAGE" = "setup" ] || [ ! -d "$HERE/dropout_prediction" ]; then
    [ -d "$HERE/dropout_prediction" ] || git clone https://github.com/wzfhaha/dropout_prediction "$HERE/dropout_prediction"
    git -C "$HERE/dropout_prediction" reset --hard a99db2e
    git -C "$HERE/dropout_prediction" apply "$HERE/cfin_changes.patch"
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$HERE/dropout_prediction"
[ -f "$X_DIR/${X_NAME}_s1.npz" ] || { echo "Missing $X_DIR/${X_NAME}_s1.npz; run 2_preprocess.py and 3_features.py first"; exit 1; }

case "$STAGE" in
smoke)
    mkdir -p my_model "$OUT/smoke"
    "$PY" main.py --seed 1 --x-dir "$X_DIR" --x-name "$X_NAME" --max-epoch 1 --smoke-rows 4000 --out-dir "$OUT/smoke" 2>&1 | tee "$OUT/smoke/main.log"
    ;;
full)
    mkdir -p my_model "$OUT"
    for seed in $SEEDS; do
        "$PY" main.py --seed "$seed" --x-dir "$X_DIR" --x-name "$X_NAME" --out-dir "$OUT" 2>&1 | tee "$OUT/main_seed$seed.log"
    done
    ;;
*)
    echo "usage: run_cfin.sh setup|smoke|full"; exit 1
    ;;
esac
