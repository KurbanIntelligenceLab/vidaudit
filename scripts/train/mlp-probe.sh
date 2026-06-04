#!/usr/bin/env bash
# Default training recipe for MLP-Probe (the standardized trainable readout).
#
#   scripts/train/mlp-probe.sh <features.csv> [--set k=v ...]
#   OUT=runs/probe scripts/train/mlp-probe.sh features/train.csv --set lr=3e-4 --set epochs=100
#
# Every value below is a default; override any of them by appending `--set key=value`
# (repeatable, later wins). Heavy training belongs on the cluster: wrap this call in
# an sbatch (see scripts/train/README.md), not a login node / your laptop.
set -euo pipefail

FEATURES="${1:?usage: mlp-probe.sh <features.csv> [--set k=v ...]}"; shift
OUT="${OUT:-runs/mlp-probe}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python run.py train mlp-probe --features "$FEATURES" --out "$OUT" \
  --set head=mlp \
  --set hidden=256,128 \
  --set dropout=0.1 \
  --set loss=bce \
  --set optimizer=adamw \
  --set lr=1e-3 \
  --set weight_decay=1e-4 \
  --set scheduler=cosine \
  --set epochs=50 \
  --set batch_size=256 \
  --set seed=42 \
  "$@"
