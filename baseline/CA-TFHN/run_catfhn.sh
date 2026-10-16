# Run the original CA-TFHN code (github.com/codeds27/CA-TFHN, commit f8ef659) on X59, activity only
# (docs/PLAN_X59.md). The changes to the original are in catfhn_changes.patch.
#
# Environment (Linux server, Python 3.9, versions of the original README):
#   conda create -n catfhn python=3.9 -y && conda activate catfhn
#   pip install -r baseline/CA-TFHN/requirements.txt
# Run from the project root, after src/2_preprocess.py and src/3_features.py (x59_s<seed>.npz):
#   bash baseline/CA-TFHN/run_catfhn.sh setup      # clone, reset to the pinned commit, apply the patch (rerun after git pull)
#   bash baseline/CA-TFHN/run_catfhn.sh features   # per seed: X59 tables, mapping, link prediction, graphs
#   bash baseline/CA-TFHN/run_catfhn.sh smoke      # seed 1, 1 epoch of 20 batches
#   bash baseline/CA-TFHN/run_catfhn.sh full       # 5 seeds -> outputs/baselines/CA-TFHN/
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
X_DIR="$ROOT/data/processed/simple"
OUT="$ROOT/outputs/baselines/CA-TFHN"
CODE="$HERE/code"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
export CATFHN_X="$X_DIR"
export CATFHN_DATA="$CODE/datastore"
STAGE="${1:-full}"

# 1. Original code at the pinned commit, plus our changes. setup always resets the clone to the
#    commit and applies the current patch, so a clone left from an older patch is never used.
if [ "$STAGE" = "setup" ] || [ ! -d "$CODE" ]; then
    [ -d "$CODE" ] || git clone https://github.com/codeds27/CA-TFHN "$CODE"
    git -C "$CODE" reset --hard f8ef659
    git -C "$CODE" apply "$HERE/catfhn_changes.patch"
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$CODE"
[ -f "$X_DIR/x59_s1.npz" ] || { echo "Missing $X_DIR/x59_s1.npz; run src/2_preprocess.py and src/3_features.py first"; exit 1; }

case "$STAGE" in
features)
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
