#!/usr/bin/env bash
# Regenerate the audited leaderboard for the wrapped detectors, end to end:
#   prepare-data (preprocess)  ->  extract (per detector)  ->  eval (audit)  ->  leaderboard
#
# Usage:
#   scripts/run_all.sh <prepared_manifest.csv>
# where <prepared_manifest.csv> is the output of
#   python run.py prepare-data <dataset> --source <your download> --out <dir>
#
# The deep backbones make extraction a GPU job -- run this on a GPU node (wrap it in
# your own sbatch). Checkpoints for the published-weight detectors are optional, via
# env vars: WAVEREP_CKPT, NSGVD_CKPT, ADM_CKPT. Output dir via OUT (default: results/).
#
# This regenerates the uniform matched-harness readout (features -> median-impute ->
# z-score -> L2-LR, LOGO + RvR) for every detector, which is the audit's apples-to-apples
# comparison. The leaderboard's native-head rows (e.g. D3, WaveRep) additionally use the
# `extract --kind score` -> `eval --scores` path shown in the README tutorial.
set -euo pipefail
MANIFEST="${1:?usage: run_all.sh <prepared_manifest.csv> (from run.py prepare-data)}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-results}"; FEAT="$OUT/features"; mkdir -p "$FEAT"

audit () {                              # $1 = detector name; $2.. = extra `extract` args
  local det="$1"; shift
  echo "=== $det: extract -> eval ==="
  python run.py extract "$det" --manifest "$MANIFEST" --out "$FEAT/$det.csv" "$@"
  python run.py eval --features "$FEAT/$det.csv" > "$OUT/$det.audit.json"
}

# clone-and-run detectors (auto-download backbones / codec MVs; no checkpoint needed):
for det in temporalspec d3 restrav clip raft fvmd; do audit "$det"; done

# published-checkpoint detectors (run only if you provide the checkpoint paths):
[ -n "${WAVEREP_CKPT:-}" ] && audit waverep --weights "$WAVEREP_CKPT"
[ -n "${NSGVD_CKPT:-}" ] && [ -n "${ADM_CKPT:-}" ] && audit nsgvd --weights "$NSGVD_CKPT" --adm-ckpt "$ADM_CKPT"

echo "=== reproduced audited metrics (uniform L2-LR readout) ==="
python - "$OUT" <<'PY'
import json, glob, os, sys
out = sys.argv[1]
print(f"{'detector':<14}{'LOGO-OOD':>9}{'RvR':>7}{'margin':>8}{'TPR@0.1%':>9}")
for f in sorted(glob.glob(os.path.join(out, "*.audit.json"))):
    d = json.load(open(f)); name = os.path.basename(f).split(".")[0]
    g = lambda k: ("%.3f" % d[k]) if d.get(k) is not None else "—"
    print(f"{name:<14}{g('logo_ood'):>9}{g('rvr'):>7}{g('margin'):>8}{g('tpr01'):>9}")
PY

echo "=== render the curated table -> LEADERBOARD.md ==="
python run.py leaderboard
