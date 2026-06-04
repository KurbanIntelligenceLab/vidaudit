#!/usr/bin/env bash
# Build VidAudit's own conda env on an HPC cluster, ON A LOGIN NODE (it has internet;
# compute nodes often do not), onto PROJECT/scratch storage (a small HOME quota fills
# up -- the package cache + the ~8 GB CUDA env must not live under HOME).
#
# Point VIDAUDIT_REPO at your checkout on the cluster:
#   ssh <cluster> 'VIDAUDIT_REPO=/path/to/vidaudit bash $VIDAUDIT_REPO/scripts/cluster_build_env.sh'
#
# Two subtleties this handles:
#  1) The login node has no GPU, so the __cuda virtual package is absent and conda
#     would resolve the *cpu* build of pytorch. CONDA_OVERRIDE_CUDA supplies __cuda
#     so the CUDA build is installable.
#  2) Even with __cuda present, conda treats cpu and cuda builds as equal candidates
#     and may still pick cpu. So we PIN the build explicitly (pytorch=*=cudaNNN*).
# conda-forge ships pytorch 2.10 for py3.14 only as CUDA 12.9 (cuda129_*), so the
# default override is 12.9 (runs on any CUDA-12.x driver via minor-version compat;
# the GPU job then asserts torch.cuda.is_available() on a real GPU). Override with
# CUDA_OVERRIDE=12.x / CUDA_BUILD=cudaNNN if conda-forge's build matrix changes.
set -euo pipefail
export PS1="${PS1:-}"
module load conda

REPO="${VIDAUDIT_REPO:?set VIDAUDIT_REPO to your vidaudit checkout on the cluster}"
PREFIX="$REPO/.conda/envs/vidaudit"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"     # package + repodata cache off HOME
export CONDA_OVERRIDE_CUDA="${CUDA_OVERRIDE:-12.9}"
CUDA_BUILD="${CUDA_BUILD:-cuda129}"
mkdir -p "$CONDA_PKGS_DIRS"
cd "$REPO"

echo "=== [1/4] create $PREFIX from environment.yml (CUDA __cuda=$CONDA_OVERRIDE_CUDA) ==="
conda env create -p "$PREFIX" -f environment.yml \
  || conda env update -p "$PREFIX" -f environment.yml --prune

echo "=== [2/4] force the CUDA build (so it is never the cpu fallback) ==="
conda install -p "$PREFIX" -c conda-forge -y "pytorch=*=${CUDA_BUILD}*"

echo "=== [3/4] editable install of vidaudit ==="
conda run -p "$PREFIX" pip install -e .

echo "=== [4/4] verify torch is the CUDA build (login node has no GPU -> is_available()=False here) ==="
conda run -p "$PREFIX" python - <<'PY'
import torch
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda)
assert torch.version.cuda is not None, "CPU build resolved -- check CUDA_OVERRIDE / CUDA_BUILD"
print("OK: CUDA-compiled torch. cuda.is_available() will be True on a GPU node.")
PY
echo "=== DONE: env ready -> regenerate the leaderboard with scripts/run_all.sh ==="
