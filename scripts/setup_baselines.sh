#!/usr/bin/env bash
# Prepare the published baselines (GĐ2) once per machine:
#   1. the original repositories in baseline/<name>, at the commit the adapters
#      src/11-14 were written for (baseline/ is not in git)
#   2. one virtual environment per baseline that needs other libraries:
#        .venvs/signet   torch 2.4 + DGL 2.4         src/12_signet.py
#        .venvs/mstgcn   torch 2.4 + DGL 2.4         src/13_mstgcn.py
#        .venvs/catfhn   torch 2.9 + PyG             src/14_catfhn.py
#      DGL's last wheels are built for torch 2.4 (CUDA 12.4). If the GPU is too new
#      for torch 2.4 ("no kernel image is available"), the check at the end says
#      so; then install DGL for a newer torch (conda channel dglteam, or from
#      source) into .venvs/signet and .venvs/mstgcn, or run them with --device cpu.
#      HyperGCN (src/11) and CFIN (src/15) need nothing more than the main .venv.
#   3. a check that every environment imports its libraries and sees the GPU
#
#   bash scripts/setup_baselines.sh                    everything
#   ONLY="catfhn" bash scripts/setup_baselines.sh      one environment
#   PYTHON_BASE=/path/to/python3.10 bash scripts/setup_baselines.sh
#
# The requirements are in scripts/venvs/<name>.txt. CFIN's original code
# (github.com/wzfhaha/dropout_prediction) is TensorFlow 1; src/15_cfin.py is a
# PyTorch port and does not need the repository.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BASE="${PYTHON_BASE:-$(command -v python3.10 || command -v python3)}"
ONLY="${ONLY:-signet mstgcn catfhn}"

# name  url  commit
REPOSITORIES=(
    "SIG-Net  https://github.com/Noverse0/SIG-Net.git      87bc320f04cc949347515d5f4c671a152e481e7e"
    "MST-GCN  https://github.com/wudongze9/MST-GCN.git     7350a7e9f3e9f042cc393997ade5921a2863c990"
    "CA-TFHN  https://github.com/codeds27/CA-TFHN.git      f8ef659450458085603652e822285c6f2be5afca"
    "HyperGCN https://github.com/malllabiisc/HyperGCN.git  de049387d8d8c9fe6ebab83cb4b49cacfd4ed534"
    "HGNN     https://github.com/iMoonLab/HGNN.git         6212e8a43e75105b7fda7b169fc0063070861bb5"
)

say() {
    echo "===== [$(date '+%d/%m/%Y %H:%M:%S')] $*"
}

mkdir -p baseline
for entry in "${REPOSITORIES[@]}"; do
    read -r name url commit <<< "$entry"
    target="baseline/$name"
    if [[ ! -d "$target/.git" ]]; then
        say "cloning $url into $target"
        git clone --quiet "$url" "$target"
    fi
    if [[ "$(git -C "$target" rev-parse HEAD)" != "$commit" ]]; then
        say "$target: checking out $commit"
        git -C "$target" fetch --quiet origin "$commit" 2>/dev/null || git -C "$target" fetch --quiet origin
        git -C "$target" checkout --quiet "$commit"
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
        say "WARNING: $target has local changes; the adapters expect the original files"
    fi
    say "$target at ${commit:0:7}"
done

say "base Python for the environments: $PYTHON_BASE ($("$PYTHON_BASE" --version 2>&1))"
for name in $ONLY; do
    venv=".venvs/$name"
    if [[ ! -x "$venv/bin/python" ]]; then
        say "creating $venv"
        "$PYTHON_BASE" -m venv "$venv"
    fi
    say "installing scripts/venvs/$name.txt into $venv"
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -r "scripts/venvs/$name.txt"
    case "$name" in
        signet|mstgcn) library="dgl" ;;
        catfhn) library="torch_geometric" ;;
        *) library="torch" ;;
    esac
    "$venv/bin/python" -c "
import $library, torch, numpy, sklearn
print('$venv: python', __import__('sys').version.split()[0], '| torch', torch.__version__,
      '| cuda', torch.cuda.is_available(), '| $library', $library.__version__)
if torch.cuda.is_available():
    try:
        (torch.ones(2, device='cuda') * 2).sum().item()
        print('$venv: GPU', torch.cuda.get_device_name(0), 'works')
    except RuntimeError as error:
        print('$venv: WARNING, torch', torch.__version__, 'cannot run on', torch.cuda.get_device_name(0), '-', error)
"
done

say "done. Run e.g.: ONLY=\"B-CFIN B-SIGNet\" bash scripts/run_scenarios.sh"
