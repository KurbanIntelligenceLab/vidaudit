#!/usr/bin/env bash
# ReStraV training recipe: extract the 21-d perceptual-straightening features from
# clips, then train a head over them via the standard trainer. The published
# features->audit L2-LR readout stays the leaderboard row; this is the retrain path.
#
#   scripts/train/restrav.sh <clips_manifest.csv> [--set k=v ...]
#   OUT=runs/restrav FEATS=feats/restrav.csv scripts/train/restrav.sh clips.csv --set lr=3e-4
#
# Feature extraction runs DINOv2 over every clip -> a GPU/cluster job for a real
# dataset (wrap in sbatch; see scripts/train/README.md). The head-train is light.
set -euo pipefail
CLIPS="${1:?usage: restrav.sh <clips_manifest.csv> [--set k=v ...]}"; shift
OUT="${OUT:-runs/restrav}"
FEATS="${FEATS:-$OUT/restrav_features.csv}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT"

# 1) extract ReStraV features from the clips (skip if the table already exists)
[ -f "$FEATS" ] || python run.py extract restrav --manifest "$CLIPS" --out "$FEATS"

# 2) train the head over the features (defaults below; override with trailing --set)
python run.py train restrav --features "$FEATS" --out "$OUT" \
  --set head=mlp \
  --set hidden=64 \
  --set dropout=0.1 \
  --set loss=bce \
  --set optimizer=adamw \
  --set lr=1e-3 \
  --set scheduler=cosine \
  --set epochs=80 \
  --set batch_size=256 \
  --set seed=42 \
  "$@"
