#!/bin/bash
#SBATCH --job-name=vidaudit_nsgvd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -A r00432
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=meocakir@iu.edu
#
# Smoke-test the NSG-VD wrapper end-to-end on a GPU. The env is built ONCE on the
# login node (scripts/cluster_build_env.sh) onto project storage -- compute nodes
# have no internet and HOME quota is too small for an ~8 GB CUDA env. This job only
# ACTIVATES that prebuilt env and verifies the GPU is live before running.
set -uo pipefail
export PS1="${PS1:-}"
module load conda

REPO=/N/project/de_briujn_graph/Projects/vidaudit
MVC=/N/project/de_briujn_graph/Projects/mv_vid_classify
PREFIX="$REPO/.conda/envs/vidaudit"
export CONDA_PKGS_DIRS="$REPO/.conda/pkgs"
cd "$REPO"

echo "=== [1/4] activate prebuilt env + verify CUDA is live on this GPU node ==="
if [ ! -d "$PREFIX" ]; then
  echo "ERROR: env missing at $PREFIX -- build it first on the login node:"
  echo "  ssh bigred 'bash $REPO/scripts/cluster_build_env.sh'"; exit 1
fi
conda run -p "$PREFIX" python - <<'PY' || exit 1
import torch
print("TORCH", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA at runtime -- check the GPU build / driver"
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== [2/4] tiny manifest of a few matched-cell clips ==="
head -n 4 "$MVC/Paper/Results/baseline_clip_subset.csv" > few.csv
wc -l few.csv

echo "=== [3/4] NSG-VD extract on GPU (ADM diffusion score -> NSG -> Swin 300-d) ==="
conda run -p "$PREFIX" python run.py extract nsgvd --manifest few.csv \
  --weights "$MVC/Baselines/nsgvd/ckpts/standard-Pika-mp.pth" \
  --adm-ckpt "$MVC/Baselines/Checkpoints/256x256_diffusion_uncond.pt" \
  --out nsgvd_smoke.csv

echo "=== [4/4] RESULT ==="
conda run -p "$PREFIX" python -c "import pandas as pd; d=pd.read_csv('nsgvd_smoke.csv'); print('rows', len(d), '| nsgvd_ cols', sum(c.startswith('nsgvd_') for c in d.columns)); print('sample', d.filter(like='nsgvd_').iloc[0][:5].round(4).tolist() if len(d) else 'NONE')"
echo "=== DONE ==="
