# Run the original MST-GCN code (github.com/wudongze9/MST-GCN, commit 7350a7e, src_xuet) on the events
# and X59 of the main model M (docs/PLAN_X59.md). The changes to the original are in mstgcn_changes.patch.
#
# Environment (Linux server; torch + DGL with the CUDA of the server):
#   conda create -n graph python=3.9 -y && conda activate graph
#   pip install -r baseline/MST-GCN/requirements.txt   # then DGL, see the comment in that file
# Run from the project root, after src/2_preprocess.py and src/3_features.py:
#   bash baseline/MST-GCN/run_mstgcn.sh setup      # clone, reset to the pinned commit, apply the patch (rerun after git pull)
#   bash baseline/MST-GCN/run_mstgcn.sh features   # rebuild the graph cache (it does not depend on the seed)
#   bash baseline/MST-GCN/run_mstgcn.sh smoke      # seed 1, 1 epoch on a 1% subset (own cache)
#   bash baseline/MST-GCN/run_mstgcn.sh full       # 5 seeds -> outputs/baselines/MST-GCN/
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
X_DIR="$ROOT/data/processed/simple"
OUT="$ROOT/outputs/baselines/MST-GCN"
CODE="$HERE/code"
CACHE="$CODE/cache"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
STAGE="${1:-full}"
# Original defaults (hidden 128, 2 layers, lr 1e-4, batch 256, dropout 0.3, 8 epochs); last epoch scored.
ARGS=(--dataset xuetangx --raw_dir "$X_DIR" --x_dir "$X_DIR")

# 1. Original code at the pinned commit, plus our changes. setup always resets the clone to the
#    commit and applies the current patch, so a clone left from an older patch is never used.
if [ "$STAGE" = "setup" ] || [ ! -d "$CODE" ]; then
    [ -d "$CODE" ] || git clone https://github.com/wudongze9/MST-GCN "$CODE"
    git -C "$CODE" reset --hard 7350a7e
    git -C "$CODE" apply "$HERE/mstgcn_changes.patch"
    [ "$STAGE" = "setup" ] && exit 0
fi
cd "$CODE/MST-GCN/src_xuet"
[ -f "$X_DIR/x59_s1.npz" ] || { echo "Missing $X_DIR/x59_s1.npz; run src/2_preprocess.py and src/3_features.py first"; exit 1; }

case "$STAGE" in
features)
    rm -rf "$CACHE/full"                                   # never reuse a graph cache built from older data
    mkdir -p "$CACHE/full" "$OUT"
    "$PY" -c "
from xuetangx_graph_builder import XuetangXGraph
XuetangXGraph(raw_dir='$X_DIR', save_dir='$CACHE/full', split_path='$X_DIR/x59_s1.npz')
" 2>&1 | tee "$CACHE/full/features.log"
    ;;
smoke)
    rm -rf "$CACHE/smoke"
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
