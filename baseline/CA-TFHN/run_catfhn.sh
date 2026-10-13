# Run the original CA-TFHN code (github.com/codeds27/CA-TFHN, commit f8ef659) on the raw data and
# learner splits of baseline/prepare_raw.py (docs/PLAN_BASELINE_v3.md). The changes to the original
# are in catfhn_changes.patch.
#
# Environment (Linux server, Python 3.9, versions of the original README):
#   conda create -n catfhn python=3.9 -y && conda activate catfhn
#   pip install -r baseline/CA-TFHN/requirements.txt
# Run from the project root, after `python baseline/prepare_raw.py`:
#   bash baseline/CA-TFHN/run_catfhn.sh setup      # clone + patch (once)
#   bash baseline/CA-TFHN/run_catfhn.sh features   # step 0 once; steps 1, 3, link prediction, graphs per split
#   bash baseline/CA-TFHN/run_catfhn.sh smoke      # split seed 1, 1 epoch of 20 batches
#   bash baseline/CA-TFHN/run_catfhn.sh full       # 5 learner splits (seed = split seed) -> outputs/baselines/CA-TFHN/
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RAW="$ROOT/data/processed/baseline_raw"
OUT="$ROOT/outputs/baselines/CA-TFHN"
CODE="$HERE/code"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
export CATFHN_RAW="$RAW"
export CATFHN_DATA="$CODE/datastore"
export CATFHN_CLUSTER="$CODE/datastore/cluster"
STAGE="${1:-full}"

# 1. Original code at the pinned commit, plus our changes.
if [ "$STAGE" = "setup" ] || [ ! -d "$CODE" ]; then
    if [ ! -d "$CODE" ]; then
        git clone https://github.com/codeds27/CA-TFHN "$CODE"
        git -C "$CODE" checkout f8ef659
        git -C "$CODE" apply "$HERE/catfhn_changes.patch"
    fi
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$CODE"
[ -f "$RAW/prediction_log/log.csv" ] || { echo "Missing $RAW; run python baseline/prepare_raw.py first"; exit 1; }

case "$STAGE" in
features)
    "$PY" src/dataprocess/0_process_user_activity_logs.py 2>&1 | tee datastore/step0.log      # all_features_35.csv
    for seed in $SEEDS; do                                                                     # datastore/s<seed>/
        mkdir -p "datastore/s$seed"
        {
            "$PY" src/dataprocess/1_process_user_contextual_features.py --split-seed "$seed"
            "$PY" src/dataprocess/3_mapping.py --split-seed "$seed"
            "$PY" src/linkprediction/linkprediction_models.py --split-seed "$seed"
            "$PY" src/graphgeneration/0_create_strong_classmates_graph.py --split-seed "$seed"
            "$PY" src/graphgeneration/1_graph_addition_information.py --split-seed "$seed"
        } 2>&1 | tee "datastore/s$seed/features.log"
    done
    ;;
smoke)
    mkdir -p "$OUT/smoke"
    "$PY" train.py -r 1 -e 1 -smoke-batches 20 -outdir "$OUT/smoke" 2>&1 | tee "$OUT/smoke/train.log"
    ;;
full)
    mkdir -p "$OUT"
    for seed in $SEEDS; do
        "$PY" train.py -r "$seed" -outdir "$OUT" 2>&1 | tee "$OUT/train_seed$seed.log"
    done
    ;;
*)
    echo "usage: run_catfhn.sh setup|features|smoke|full"; exit 1
    ;;
esac
