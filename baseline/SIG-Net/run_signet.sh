# Run the original SIG-Net model (github.com/Noverse0/SIG-Net, commit 87bc320) on the raw data and
# learner splits of baseline/prepare_raw.py (docs/PLAN_BASELINE_v3.md). The original only has KDD Cup
# and NAVER; code/src/xuetangx_graph_builder.py translates kddcup_graph to XuetangX. All changes are
# in signet_changes.patch.
#
# Environment: the one of MST-GCN (baseline/MST-GCN/requirements.txt + DGL).
# Run from the project root, after `python baseline/prepare_raw.py`:
#   bash baseline/SIG-Net/run_signet.sh setup      # clone + patch (once)
#   bash baseline/SIG-Net/run_signet.sh check      # vectorized builder = loop translation on a sample (no DGL needed)
#   bash baseline/SIG-Net/run_signet.sh smoke      # split seed 1, 1 epoch on the first 2,000 enrollments of every split
#   bash baseline/SIG-Net/run_signet.sh full       # 5 learner splits (seed = split seed) -> outputs/baselines/SIG-Net/
# The graph depends on the split (train-only scaler, label types of train sources), so every run
# builds it again (a few minutes).
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RAW="$ROOT/data/processed/baseline_raw"
OUT="$ROOT/outputs/baselines/SIG-Net"
CODE="$HERE/code"
SEEDS="${SEEDS:-1 11 111 1111 11111}"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
STAGE="${1:-full}"
# Original defaults (RGCN, hidden 64, 2 layers, lr 1e-3, batch 256, dropout 0.3); at most 3 x 4 epochs.
ARGS=(--dataset xuetangx --raw_dir "$RAW" --split_dir "$RAW/splits" --num_epochs 12 --patience 5)

# 1. Original code at the pinned commit, plus our changes.
if [ "$STAGE" = "setup" ] || [ ! -d "$CODE" ]; then
    if [ ! -d "$CODE" ]; then
        git clone https://github.com/Noverse0/SIG-Net "$CODE"
        git -C "$CODE" checkout 87bc320
        git -C "$CODE" apply "$HERE/signet_changes.patch"
    fi
    [ "$STAGE" = "setup" ] && exit 0
fi
[ -f "$RAW/prediction_log/log.csv" ] || { echo "Missing $RAW; run python baseline/prepare_raw.py first"; exit 1; }

case "$STAGE" in
check)
    "$PY" "$HERE/check_builder.py" --raw-dir "$RAW"
    ;;
smoke)
    mkdir -p "$OUT/smoke"
    cd "$CODE"
    "$PY" src/main.py "${ARGS[@]}" --seed 1 --num_epochs 1 --smoke 2000 --out_dir "$OUT/smoke" --log_dir "$OUT/smoke"
    ;;
full)
    mkdir -p "$OUT/logs"
    cd "$CODE"
    for seed in $SEEDS; do
        "$PY" src/main.py "${ARGS[@]}" --seed "$seed" --out_dir "$OUT" --log_dir "$OUT/logs"
    done
    ;;
*)
    echo "usage: run_signet.sh setup|check|smoke|full"; exit 1
    ;;
esac
