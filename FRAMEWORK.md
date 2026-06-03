# vidaudit — audited evaluation + training toolkit and leaderboard for AI-generated video detection

> Working name; rebrand TBD (you + advisor).

## Goal
One repo a newcomer can clone to:
1. **Evaluate** any detector through a standardized **audited** protocol (P1–P6) and the audited metric tuple, with one command.
2. **Train / retrain** detectors that ship training code, through a uniform script on the standardized data.
3. **Add their own method** behind a small **plugin API** and land on a shared, audited leaderboard.

Backed by a standardized **data package** (features + splits + provenance; bring-your-own-videos) and a **model zoo** of published-weight baselines.

## Why
Every AIGV-detection paper currently evaluates differently; the 20-paper survey shows none apply all six controls; and high leaderboard AUCs do not predict deployable recall. This toolkit makes the audited protocol the default and makes fair, reproducible comparison (and honest re-ranking by deployability) a single command.

## Repo layout
```
vidaudit/
  vidaudit/
    detectors/        # the plugin API + one wrapper per method
      base.py         # Detector ABC + DetectorSpec  (the plugin contract)
      registry.py     # @register / get / all_detectors
      temporalspec.py  d3.py  restrav.py  waverep.py  nsgvd.py  fvmd.py  raft.py  clip.py
    data/
      canonical.py    # P1: canonical H.264 re-encode
      filters.py      # P2: K-non-I-frame leakage filter
      cells.py        # matched cells, LOGO splits, provenance (GenVidBench / AIGVDBench / combined)
      fetch.py        # bring-your-own-videos: fetch + verify from source (no redistribution)
    audit/
      protocol.py     # P1–P6 orchestration; one audited record per detector
      metrics.py      # AUC, pAUC@10%, TPR@1%, TPR@0.1%, Brier, ECE, RvR, margin, bootstrap CIs
      verdict.py      # certified / caught(dataset-identity) / collapses(operating-point) / leakage
      leaderboard.py  # write/update leaderboard.csv (+ Croissant)
    train/
      trainer.py      # standardized training interface; wraps authors' recipes where released
  run.py              # eval | train | leaderboard  (single entry point)
  configs/            # per-detector + per-cell configs
  leaderboard.csv
  README.md  FRAMEWORK.md
```

## Plugin contract (how a newcomer adds a method)
Subclass `Detector`, set a `DetectorSpec`, implement **at least one** of:
- `score(clip) -> p(generated)` — the native head (what the leaderboard scores), or
- `features(clip) -> vector` — optional; fed to the uniform matched-harness L2-LR readout.

Optionally implement `train(data, out_dir)` to be retrainable through `run.py train`. Decorate with `@register("name")`. Nothing else — the audit harness handles P1–P6, the metrics, the verdict, and the leaderboard row.

## Eval vs. train (scope discipline)
- **Eval (primary):** we run the **published weights / native heads** through the audit. This populates the leaderboard and is what surfaces any failing model (high AUC but collapses at a deployable threshold, or rides dataset identity).
- **Train (provided, not reproduced):** a uniform `train()` wrapper for methods whose authors released training code (ReStraV, NSG-VD, WaveRep, TemporalSpec). We expose the standardized interface + standardized data; we do **not** promise to reproduce every paper's headline number — that's the user's experiment.

## Data package (HuggingFace)
We **cannot redistribute** GenVidBench / AIGVDBench videos (copyright). The released "dataset" is: per-clip **features** for each wrapped detector, the canonical-re-encode + K-filter **recipe**, the matched-cell **membership + LOGO splits + provenance labels**, and **Croissant** metadata. `data/fetch.py` reconstructs clips from the original sources on the user's machine. A **combined GenVidBench+AIGVDBench** audited cell is provided with source tags.

## Model zoo (seed roster; published weights for eval)
| detector | backbone | weights | train code | role |
|---|---|---|---|---|
| TemporalSpec (ours) | codec motion vectors (13-d) | ours | yes | white-box control |
| D3 | XCLIP-ViT-B/16 | training-free | n/a | published detector |
| ReStraV | DINOv2 ViT-S/14 | public | public | published detector |
| WaveRep | DINOv2 ViT-B/14 + wavelet aug | public | partial | published detector |
| NSG-VD | per nsgvd repo | public (ckpts) | public | published detector (audit pending) |
| FVMD | PIPs++ point-tracker | metric | n/a | repurposed reference |
| RAFT | RAFT-Large flow | pretrained | n/a | repurposed reference |
| CLIP | CLIP-ViT-B/32 | pretrained | n/a | repurposed reference |

Excluded for now (no public weights): **DeMamba** (authors withhold, GitHub issues #5/#16/#21), DeCoF / ATSS / CMTA / VidGuard-R1 (no release) — kept on a "wanted: weights" list.

## Leaderboard schema
`model, backbone, weights?, cell, LOGO-ID, LOGO-OOD, RvR, margin, pAUC@10%, TPR@1%, TPR@0.1%, Brier, ECE, cost_ms, verdict, source`

## Build phases (engineering order — not deadline-driven)
1. **Core** — `base.py`, `registry.py`, `run.py`, `audit/metrics.py` + `verdict.py`, `data/cells.py`, reusing the existing matched-cell + LOGO + metric code (`Baselines/evaluate_features.py`, `run_audited_metrics.py`).
2. **Eval wrappers** for the seed roster (reuse `Baselines/run_*_features.py` logic). Each yields a leaderboard row — this also surfaces failing models.
3. **Data layer** — `canonical.py` + `filters.py` + `fetch.py` + the standardized data package + Croissant (HF dataset).
4. **Training** — `train/trainer.py` + per-detector wrappers (ReStraV, NSG-VD, WaveRep, TemporalSpec).
5. **Release** — leaderboard rendering + HF Space + README + a "add your model in 20 lines" tutorial.

## Status
- [ ] Phase 1 core (in progress: plugin API drafted)
- [ ] Phase 2 eval wrappers (D3-native + NSG-VD audit running on the cluster — feeds the first rows)
- [ ] Phases 3–5
