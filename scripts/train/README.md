# Training recipes

Each `scripts/train/<model>.sh` encodes a detector's **default** hyperparameters and
calls the standard trainer. Override any knob with `--set key=value` (repeatable;
later wins) without editing the script:

```bash
# defaults
scripts/train/mlp-probe.sh features/train.csv

# override loss, lr, schedule; write elsewhere
OUT=runs/probe-focal scripts/train/mlp-probe.sh features/train.csv \
  --set loss=focal --set lr=3e-4 --set scheduler=step --set epochs=100
```

Overridable `TrainConfig` fields: `head` (`mlp`|`linear`), `hidden` (comma list),
`dropout`, `loss` (`bce`|`focal`), `optimizer` (`adamw`|`adam`|`sgd`), `lr`,
`weight_decay`, `momentum`, `scheduler` (`cosine`|`step`|`none`), `epochs`,
`batch_size`, `amp`, `grad_clip`, `seed`, `device`, `val_frac`, `feature_cols`,
`subset`, `val_features`. Unknown keys land in `cfg.extra` (e.g. `--set focal_gamma=1.5`).
Loss / optimizer / scheduler / head are name-addressable registries
(`vidaudit/train/registries.py`) — add one with a decorator and it is immediately
usable from `--set`.

The trainer learns a head over a **precomputed feature table** (an `extract`
output), so extract once and reuse:

```bash
python run.py extract temporalspec --manifest clips.csv --out features/train.csv
scripts/train/mlp-probe.sh features/train.csv
```

It writes `<out>/model.pt` (state dict + config + feature columns + the persisted
median-impute/z-score stats + best val AUC) and `<out>/metrics.json` (per-epoch
history). The official numbers still come from the audit (`run.py eval`); the
trainer's internal val AUC only selects the checkpoint.

## Run heavy training on the cluster, never locally

Wrap the recipe in an sbatch (see `reference-cluster` for the template):

```bash
mkdir -p logs && sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=vidaudit_train --partition=gpu --gpus=1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00
#SBATCH -A r00432 --output=logs/%x_%j.out --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL --mail-user=meocakir@iu.edu
export PS1="${PS1:-}"
module load conda && conda activate vidaudit
cd /N/project/de_briujn_graph/Projects/vidaudit
OUT=runs/probe scripts/train/mlp-probe.sh features/train.csv --set epochs=200
EOF
```
