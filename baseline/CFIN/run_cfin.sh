# Run the original CFIN code (github.com/wzfhaha/dropout_prediction, commit a99db2e) on the raw
# data and learner splits of baseline/prepare_raw.py (docs/PLAN_BASELINE_v3.md). The changes to
# the original are in cfin_changes.patch.
#
# Environment (Python 3.7, TensorFlow 1.15; Linux server):
#   conda create -n cfin python=3.7 -y && conda activate cfin
#   pip install -r baseline/CFIN/requirements.txt
# Run from the project root, after `python baseline/prepare_raw.py`:
#   bash baseline/CFIN/run_cfin.sh setup      # clone + patch (once)
#   bash baseline/CFIN/run_cfin.sh features   # feat_extract.py once + preprocess.py per learner split
#   bash baseline/CFIN/run_cfin.sh smoke      # split seed 1, 1 epoch on the first 4,000 rows of every split
#   bash baseline/CFIN/run_cfin.sh full       # 5 learner splits (seed = split seed) -> outputs/baselines/CFIN/
# Windows (local .venv): PY=baseline/CFIN/.venv/Scripts/python.exe bash baseline/CFIN/run_cfin.sh ...
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RAW="$ROOT/data/processed/baseline_raw"
OUT="$ROOT/outputs/baselines/CFIN"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CFIN_RAW="$RAW/prediction_log"
export CFIN_SPLITS="$RAW/splits"
STAGE="${1:-full}"

# 1. Original code at the pinned commit, plus our changes.
if [ "$STAGE" = "setup" ] || [ ! -d "$HERE/dropout_prediction" ]; then
    if [ ! -d "$HERE/dropout_prediction" ]; then
        git clone https://github.com/wzfhaha/dropout_prediction "$HERE/dropout_prediction"
        git -C "$HERE/dropout_prediction" checkout a99db2e
        git -C "$HERE/dropout_prediction" apply "$HERE/cfin_changes.patch"
    fi
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$HERE/dropout_prediction"
[ -f "$CFIN_RAW/log.csv" ] || { echo "Missing $RAW; run python baseline/prepare_raw.py first"; exit 1; }

case "$STAGE" in
features)
    cp "$RAW/user_info.csv" "$RAW/course_info.csv" .
    "$PY" feat_extract.py 2>&1 | tee feat_extract.log      # all_features.csv (counts do not depend on the split)
    for seed in $SEEDS; do                                 # feat/s<seed>/{train,validation,test}_feat.csv, act_feats.pkl
        "$PY" preprocess.py --split-seed "$seed" 2>&1 | tee "preprocess_s$seed.log"
    done
    ;;
smoke)
    mkdir -p my_model "$OUT/smoke"
    "$PY" main.py --seed 1 --max-epoch 1 --smoke-rows 4000 --out-dir "$OUT/smoke" 2>&1 | tee "$OUT/smoke/main.log"
    ;;
full)
    mkdir -p my_model "$OUT"
    for seed in $SEEDS; do
        "$PY" main.py --seed "$seed" --out-dir "$OUT" 2>&1 | tee "$OUT/main_seed$seed.log"
    done
    ;;
*)
    echo "usage: run_cfin.sh setup|features|smoke|full"; exit 1
    ;;
esac
