#!/usr/bin/env bash
# Build VidAudit's own conda env on BigRed, ON THE LOGIN NODE (it has internet;
# GPU/compute nodes do not), onto PROJECT storage (HOME quota is tiny and fills up
# -- the package cache + the ~8 GB CUDA env must not live under /N/u).
#
# The login node has no GPU, so conda would normally resolve the *cpu* build of
# pytorch. We force the CUDA build by setting the __cuda virtual package via
# CONDA_OVERRIDE_CUDA; torch.version.cuda is then non-null and the GPU job
# (scripts/cluster_verify_nsgvd.sbatch) verifies torch.cuda.is_available() on a
# real GPU. Override the CUDA version with CUDA_OVERRIDE=12.x if needed.
#
#   ssh bigred 'bash /N/project/de_briujn_graph/Projects/vidaudit/scripts/cluster_build_env.sh'
set -euo pipefail
export PS1="${PS1:-}"
module load conda

REPO=/N/project/de_briujn_graph/Projects/vidaudit
PREFIX="$REPO/.conda/envs/vidaudit"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"     # package + repodata cache off HOME
mkdir -p "$CONDA_PKGS_DIRS"
cd "$REPO"

echo "=== solving + creating $PREFIX (CUDA build forced) ==="
CONDA_OVERRIDE_CUDA="${CUDA_OVERRIDE:-12.6}" \
  conda env create -p "$PREFIX" -f environment.yml --yes \
  || CONDA_OVERRIDE_CUDA="${CUDA_OVERRIDE:-12.6}" \
     conda env update -p "$PREFIX" -f environment.yml --prune

conda run -p "$PREFIX" pip install -e .

echo "=== verify torch is the CUDA build (login node has no GPU, so is_available()=False here) ==="
conda run -p "$PREFIX" python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
assert torch.version.cuda is not None, "CPU build resolved -- re-run with CUDA_OVERRIDE set"
print("OK: CUDA-compiled torch. cuda.is_available() will be True on a GPU node.")
PY
echo "=== DONE: activate with  conda activate $PREFIX ==="
