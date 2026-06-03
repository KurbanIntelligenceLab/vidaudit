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
# Build VidAudit's OWN conda env on a GPU node (so conda resolves a CUDA torch),
# then smoke-test the NSG-VD wrapper end-to-end on a few real clips.

export PS1="${PS1:-}"
module load conda

REPO=/N/project/de_briujn_graph/Projects/vidaudit
MVC=/N/project/de_briujn_graph/Projects/mv_vid_classify
cd "$REPO"

echo "=== [1/4] build/refresh the vidaudit env (GPU node -> CUDA torch) ==="
conda env create -n vidaudit -f environment.yml \
  || conda env update -n vidaudit -f environment.yml --prune
conda run -n vidaudit pip install -e .
conda run -n vidaudit python -c "import torch; print('TORCH', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "=== [2/4] tiny manifest of a few matched-cell clips ==="
head -n 4 "$MVC/Paper/Results/baseline_clip_subset.csv" > few.csv
wc -l few.csv

echo "=== [3/4] NSG-VD extract on GPU (ADM diffusion score -> NSG -> Swin 300-d) ==="
conda run -n vidaudit python run.py extract nsgvd --manifest few.csv \
  --weights "$MVC/Baselines/nsgvd/ckpts/standard-Pika-mp.pth" \
  --adm-ckpt "$MVC/Baselines/Checkpoints/256x256_diffusion_uncond.pt" \
  --out nsgvd_smoke.csv

echo "=== [4/4] RESULT ==="
conda run -n vidaudit python -c "import pandas as pd; d=pd.read_csv('nsgvd_smoke.csv'); print('rows', len(d), '| nsgvd_ cols', sum(c.startswith('nsgvd_') for c in d.columns)); print('sample', d.filter(like='nsgvd_').iloc[0][:5].round(4).tolist() if len(d) else 'NONE')"
echo "=== DONE ==="
