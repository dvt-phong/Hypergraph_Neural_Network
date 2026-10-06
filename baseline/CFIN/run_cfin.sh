# Run the original CFIN code (github.com/wzfhaha/dropout_prediction, commit a99db2e) with
# log1p + z-score on the activity counts. The changes to the original are in cfin_changes.patch.
#
# Environment (Python 3.7, TensorFlow 1.15; Linux server):
#   conda create -n cfin python=3.7 -y && conda activate cfin
#   pip install -r baseline/CFIN/requirements.txt
# Run from the project root (needs data/raw/xuetangx from src/1_download.py):
#   CUDA_VISIBLE_DEVICES=0 bash baseline/CFIN/run_cfin.sh
# Windows (local .venv): PY=baseline/CFIN/.venv/Scripts/python.exe bash baseline/CFIN/run_cfin.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$HERE/../../data/raw/xuetangx"
PY="${PY:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac   # path -> absolute (we cd below)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 1. Original code at the pinned commit, plus our changes.
if [ ! -d "$HERE/dropout_prediction" ]; then
    git clone https://github.com/wzfhaha/dropout_prediction "$HERE/dropout_prediction"
    git -C "$HERE/dropout_prediction" checkout a99db2e
    git -C "$HERE/dropout_prediction" apply "$HERE/cfin_changes.patch"
fi
cd "$HERE/dropout_prediction"

# 2. Data in the layout of dump_data.sh.
[ -d prediction_log ] || tar xzf "$RAW/prediction_data.tar.gz"
[ -f user_info.csv ] || cp "$RAW/user_info.csv" .
[ -f course_info.csv ] || cp "$RAW/course_info.csv" .

# 3. Steps of the README.
"$PY" feat_extract.py 2>&1 | tee feat_extract.log      # train_features.csv, test_features.csv
"$PY" preprocess.py 2>&1 | tee preprocess.log          # train_feat.csv, test_feat.csv, act_feats.pkl
mkdir -p feat my_model                                 # main.py reads feat/; model.fit saves to my_model/
mv train_feat.csv test_feat.csv feat/
"$PY" main.py 2>&1 | tee main.log                      # one line per epoch: valid-result = test AUC
