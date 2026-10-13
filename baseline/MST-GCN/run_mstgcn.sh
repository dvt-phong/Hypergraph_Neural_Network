# Run the original MST-GCN code (github.com/wudongze9/MST-GCN, commit 7350a7e, src_xuet) on the raw
# data and learner splits of baseline/prepare_raw.py (docs/PLAN_BASELINE_v3.md). The changes to the
# original are in mstgcn_changes.patch.
#
# Environment (Linux server; torch + DGL with the CUDA of the server, shared with SIG-Net):
#   conda create -n graph python=3.9 -y && conda activate graph
#   pip install -r baseline/MST-GCN/requirements.txt   # then DGL, see the comment in that file
# Run from the project root, after `python baseline/prepare_raw.py`:
#   bash baseline/MST-GCN/run_mstgcn.sh setup      # clone + patch (once)
#   bash baseline/MST-GCN/run_mstgcn.sh features   # build the graph cache once (it does not depend on the split)
#   bash baseline/MST-GCN/run_mstgcn.sh smoke      # split seed 1, 1 epoch on a 1% subset (own cache)
#   bash baseline/MST-GCN/run_mstgcn.sh full       # 5 learner splits (seed = split seed) -> outputs/baselines/MST-GCN/
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RAW="$ROOT/data/processed/baseline_raw"
OUT="$ROOT/outputs/baselines/MST-GCN"
CODE="$HERE/code"
CACHE="$CODE/cache"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
STAGE="${1:-full}"
# Original defaults (hidden 128, 2 layers, lr 1e-4, batch 256, dropout 0.3); at most 3 x 8 epochs.
ARGS=(--dataset xuetangx --raw_dir "$RAW/prediction_log" --split_dir "$RAW/splits" --num_epochs 24 --patience 5)

# 1. Original code at the pinned commit, plus our changes.
if [ "$STAGE" = "setup" ] || [ ! -d "$CODE" ]; then
    if [ ! -d "$CODE" ]; then
        git clone https://github.com/wudongze9/MST-GCN "$CODE"
        git -C "$CODE" checkout 7350a7e
        git -C "$CODE" apply "$HERE/mstgcn_changes.patch"
    fi
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$CODE/MST-GCN/src_xuet"
[ -f "$RAW/prediction_log/log.csv" ] || { echo "Missing $RAW; run python baseline/prepare_raw.py first"; exit 1; }

case "$STAGE" in
features)
    mkdir -p "$CACHE/full" "$OUT"
    "$PY" -c "
from xuetangx_graph_builder import XuetangXGraph
XuetangXGraph(raw_dir='$RAW/prediction_log', save_dir='$CACHE/full', split_path='$RAW/splits/learner_s1.csv')
" 2>&1 | tee "$CACHE/full/features.log"
    ;;
smoke)
    mkdir -p "$CACHE/smoke" "$OUT/smoke"
    "$PY" main.py "${ARGS[@]}" --cache_dir "$CACHE/smoke" --subset_ratio 0.01 --seed 1 --num_epochs 1 \
        --out_dir "$OUT/smoke" --log_dir "$OUT/smoke" 2>&1 | tee "$OUT/smoke/main.log"
    ;;
full)
    mkdir -p "$OUT"
    for seed in $SEEDS; do
        "$PY" main.py "${ARGS[@]}" --cache_dir "$CACHE/full" --seed "$seed" --out_dir "$OUT" --log_dir "$OUT/logs"
    done
    ;;
*)
    echo "usage: run_mstgcn.sh setup|features|smoke|full"; exit 1
    ;;
esac
