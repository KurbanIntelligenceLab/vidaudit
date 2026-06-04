<h1 align="center">VidAudit</h1>

<p align="center">
  <b>An audited benchmark, leaderboard, and toolkit for AI-generated video detection.</b><br>
  <sub>Evaluate any detector under one rigorous protocol. Train any method through a uniform script. Add your own behind a small plugin API.</sub>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Audit protocol" src="https://img.shields.io/badge/audit-P1--P6-8A2BE2.svg">
</p>

VidAudit is a standardized, **audited** evaluation suite for **AI-generated, synthetic, and deepfake video detection**. It exists because every AI-video-detection paper currently evaluates differently, a 20-paper survey shows none apply all six standard controls, and high leaderboard AUCs do not predict deployable recall. VidAudit makes the audited protocol the default and makes fair, reproducible comparison (and honest re-ranking by deployability) a single command.

---

## 🏆 Audited leaderboard

The primary cell is the **matched 27k-clip GenVidBench cell** under leave-one-generator-out (LOGO) evaluation. The headline number papers report is **LOGO-OOD AUC**; the audit adds the columns that decide whether it is real: the **real-vs-real floor** (`RvR`), the **above-floor margin**, and **deployable recall** (`TPR@0.1%`). Sorted by OOD AUC, so high-AUC methods visibly separate from genuinely robust ones.

| # | Model | LOGO-OOD AUC ↑ | RvR floor | Margin ↑ | TPR@0.1% ↑ | Verdict |
|--:|---|:--:|:--:|:--:|:--:|---|
| 1 | **WaveRep** | 0.996 | 0.534 | +0.462 | **0.816** | ✅ Certified, usable |
| 2 | **XSFF** (ours) | 0.946 | 0.604 | +0.342 | — | ✅ Certified |
| 3 | **ReStraV** | 0.931 | 0.586 | +0.345 | 0.634 | ✅ Certified, usable |
| 4 | **D3** | 0.887 | 0.421 | +0.466 | — | ✅ Certified (native head) |
| 5 | **FVMD** | 0.880 | 0.574 | +0.306 | 0.027 | ⚠️ Certified, collapses @0.1% |
| 6 | **TemporalSpec+aug** (ours) | 0.871 | 0.634 | +0.237 | — | ✅ Certified |
| 7 | **RAFT** | 0.855 | 0.627 | +0.228 | 0.020 | ⚠️ Certified, collapses @0.1% |
| 8 | **CLIP** | 0.852 | 0.766 | +0.086 | 0.238 | ❌ Caught (dataset identity) |
| 9 | **TemporalSpec** (ours) | 0.832 | 0.643 | +0.189 | 0.024 | ⚠️ Certified, collapses @0.1% |
| 10 | **NSG-VD** | 0.660 | 0.596 | +0.064 | — | ❌ Near-floor (rides identity) |

**How to read the verdict.**
- ✅ **Certified**: clears its real-vs-real floor by a wide margin, genuine cross-generator signal.
- ✅ **Certified, usable**: also keeps useful recall at a deployable false-positive rate.
- ⚠️ **Certified, collapses @0.1%**: strong AUC, but recall at FPR = 0.1% falls to near zero. The ranking and the deployability disagree.
- ❌ **Caught (dataset identity)**: a high real-vs-real AUC means the score largely reflects *which dataset the reals came from*, not generation artifacts.
- ❌ **Near-floor**: barely exceeds the dataset-identity floor at all.

**Column definitions.**
- **LOGO-OOD AUC**: leave-one-generator-out AUC on held-out generators.
- **RvR floor**: real-vs-real AUC (separating two *real* datasets). An artifact-based detector should sit near 0.5; a high value means dataset-identity leakage.
- **Margin**: LOGO-OOD minus RvR, the real generalization remaining above the floor.
- **TPR@0.1%**: true-positive rate at a 0.1% false-positive operating point (deployable recall).

**Other cells** (also in `leaderboard.csv` / `LEADERBOARD.md`).
- **AIGVDBench** (`aigvd-2284`): D3 scores **0.771** LOGO-OOD (native head). Only one real source, so the RvR floor does not apply; the current run used XCLIP-B/32 and a B/16 re-run is pending. Broader AIGVDBench coverage lands as those features ship.
- **GenVidBench full-116k** (unmatched): TemporalSpec **0.819**, for reference against the matched cell.

<sub>GenVidBench matched-27k cell; native head or uniform L2-LR readout per method. A `—` in TPR@0.1% marks an operating point not yet in the released `leaderboard.csv`. The full 116k cell, bootstrap CIs, and the combined GenVidBench + AIGVDBench cross-dataset cell land here as the data package ships.</sub>

---

## What's inside
- **Six-control audit protocol (P1-P6)**: canonical re-encode, clip-length leakage filter, real-vs-real dataset-identity floor, matched-harness re-training, multi-seed/bootstrap CIs, and a true cross-dataset cell.
- **Audited leaderboard**: every detector labeled by the two verdicts above, with the full metric tuple (AUC, above-floor margin, TPR@FPR, calibration), not just AUC.
- **Model zoo**: 8 detectors behind one plugin API; backbones auto-download or load published checkpoints (sha256-verified via the zoo).
- **Standardized data package**: per-clip features, LOGO splits, provenance, and Croissant metadata, combining GenVidBench and AIGVDBench (bring-your-own-videos; we do not redistribute source clips).
- **Unified CLI**: `run.py extract` (clips → features) → `run.py eval` (audit → verdicts) → `run.py leaderboard`, plus a uniform, overridable trainer driven by shell scripts.

## Setup

**Prerequisites:** [Conda](https://docs.conda.io/) (Miniforge or Miniconda) and `git`. A GPU is optional — Apple Silicon (MPS) and CPU both work for development; heavy extraction and training are meant for a CUDA cluster, where the same `environment.yml` resolves a GPU build of torch.

```bash
# 1. Clone the repository
git clone <repo-url> vidaudit && cd vidaudit

# 2. Create the one unified environment (Python 3.14: audit + video/codec tooling +
#    deep backbones + training, in a single env)
conda env create -f environment.yml
#    For a byte-exact environment instead of pinned ranges:
#    conda env create -f environment.lock.yml

# 3. Activate it
conda activate vidaudit

# 4. Register the package (editable install; runtime deps come from conda, not pip)
pip install -e .

# 5. Verify
python -m pytest -q       # the test suite should pass
python run.py --help      # the CLI entry point
```

That single environment covers everything. On a CUDA cluster, create the env on a node with internet so conda resolves the GPU torch build — see `scripts/cluster_build_env.sh` for a turnkey script.

## Tutorial

The workflow is **extract → audit → (train) → leaderboard**, one command per step. The input is a `clips.csv` manifest with columns `(video_id, generator, label, is_real, mp4_path)`: one row per clip, `is_real=1` for real sources and `0` for generated, and `generator` naming the model (or the real source).

**Step 1 — Extract a detector's per-clip features.** The auto-download detectors (D3, ReStraV, CLIP, RAFT) and TemporalSpec (codec motion vectors) need no setup; WaveRep / FVMD / NSG-VD take a checkpoint via `--weights`.
```bash
python run.py extract restrav --manifest clips.csv --out restrav.csv
python run.py extract waverep --manifest clips.csv --out waverep.csv \
       --weights /path/to/weights_dinov2_G4.ckpt
```

**Step 2 — Audit the feature table.** Runs the six-control protocol and prints one leaderboard record. `--subset` restricts to a matched cell.
```bash
python run.py eval --features restrav.csv --subset baseline_clip_subset.csv
```
The record reports `logo_ood` (cross-generator AUC), `rvr` (the real-vs-real floor), `margin` (above-floor headroom), and `tpr01` / `pauc10` / `brier` / `ece` — plus a **floor verdict** (certified / caught / marginal / leakage) and a **deploy tier** (usable / marginal / collapses).

**Step 3 — Audit a native head's own scores** (no readout retraining), e.g. D3's published decision:
```bash
python run.py extract d3 --manifest clips.csv --out d3_scores.csv --kind score
python run.py eval --scores d3_scores.csv
```
To train instead of using a published head, the standard trainer fits a head over any feature table — `scripts/train/restrav.sh clips.csv` or `python run.py train mlp-probe --features restrav.csv` (see [Training](#training)).

**Step 4 — Render the leaderboard** from `leaderboard.csv`:
```bash
python run.py leaderboard      # writes LEADERBOARD.md
```

Heavy extraction or training belongs on a cluster — wrap any step in an sbatch (templates in `scripts/`). `fetch-weights <name>` downloads and sha256-verifies a checkpoint from the zoo; `fetch-data` reconstructs a dataset locally (P1 re-encode + P2 length filter).

## Add your own detector (plugin API)
Subclass `Detector`, set a `DetectorSpec`, implement at least one evidence interface, and register it. The audit, metrics, and leaderboard row come for free:
```python
from vidaudit.detectors import Detector, DetectorSpec, register

@register("mymethod")
class MyMethod(Detector):
    spec = DetectorSpec(name="MyMethod", family="appearance", backbone="ViT-L/14",
                        published_weights=True, trainable=False, paper="arXiv:25xx")

    def score(self, clip):        # native head -> p(generated)
        return my_model(clip.path)

    # optional: features(clip) for the uniform L2-LR readout,
    #           load_weights() for download-and-evaluate,
    #           build_model()/default_train_config() to be trainable.
# python run.py eval mymethod   ->   P1-P6 audit + a leaderboard row
```
A detector has three orthogonal capabilities: **evaluate** (`score` and/or `features`), **load weights** (`load_weights`), and **train** (`build_model` + `default_train_config`, driven by the standard trainer). Training-free methods simply leave `trainable=False`. See `FRAMEWORK.md` for the full contract.

## Add a dataset
Datasets live behind a registry, so the audit pipeline never changes. A new benchmark is a thin adapter:
```python
from vidaudit.data.datasets import VideoDataset, DatasetSpec, register_dataset

@register_dataset("mybench")
class MyBench(VideoDataset):
    spec = DatasetSpec(name="mybench", generators=[...],
                       real_sources=["src_a", "src_b"],   # >=2 enables the RvR floor; <2 auto-disables it
                       fetch=<recipe>)                      # reconstruct clips locally (no redistribution)
    def clips(self, split, cell):
        ...   # yield Clip objects with provenance
# now available everywhere:  --dataset mybench   (and combinable:  --dataset genvidbench+mybench)
```

## Model zoo and weights
All eight baselines are wrapped behind the plugin API and run from clips via `run.py extract`. Five are clone-and-run (auto-download backbones or codec features); three take published checkpoints, fetched via the zoo (`fetch_weights` → download + sha256-verify + cache) or passed with `--weights`.

| Detector | Family | Backbone | Setup |
|---|---|---|---|
| TemporalSpec (ours) | codec | codec motion vectors (13-d) | none — PyAV codec MVs |
| D3 | appearance | XCLIP-ViT-B/16 | auto-download (training-free) |
| ReStraV | appearance | DINOv2 ViT-S/14 | auto-download (torch.hub) |
| CLIP | appearance | CLIP-ViT-B/32 | auto-download |
| RAFT | motion | RAFT-Large optical flow | auto-download (torchvision) |
| FVMD | motion | PIPs++ point tracker | auto-fetch (public release) |
| WaveRep | forensic | DINOv2 ViT-B/14 + wavelet aug | checkpoint (zoo / `--weights`) |
| NSG-VD | forensic | ADM 256 diffusion + Swin discriminator | checkpoint + ~2GB ADM model |

Vendored model code (PIPs++ for FVMD, the NSG-VD codebase) lives under `vidaudit/_vendor/`, kept separate from our thin wrappers and carrying its upstream license.

**Weight downloads.** FVMD's point tracker auto-fetches from its public release. The WaveRep, NSG-VD, and ADM checkpoints have their sha256s recorded in `zoo/manifest.yaml`; a public mirror is on the way — until then, obtain the checkpoint and pass it with `--weights`.

| Checkpoint | Size | Download |
|---|---|---|
| WaveRep `weights_dinov2_G4.ckpt` | 331 MB | _placeholder — public mirror coming_ |
| NSG-VD `standard-Pika-mp.pth` | 2 MB | _placeholder — public mirror coming_ |
| ADM `256x256_diffusion_uncond.pt` | 2.1 GB | _placeholder — public mirror coming_ |

Excluded for now (no public weights): **DeMamba** (authors withhold the checkpoints, GitHub issues #5/#16/#21), DeCoF / ATSS / CMTA / VidGuard-R1 (no release). On a "wanted: weights" list.

## Data package
We **cannot redistribute** the GenVidBench / AIGVDBench source videos (copyright). VidAudit ships the standardized **per-clip features, LOGO/RvR splits, provenance labels, and a reproducible recipe** — two controls applied to *your* local source clips so everyone evaluates byte-identical inputs:

- **P1 canonical re-encode** (`vidaudit/data/canonical.py`): one fixed H.264 recipe normalizes codec/container fingerprints (recorded as a `recipe_id` for provenance; H.264 because the codec-MV detectors need H.264 motion vectors).
- **P2 clip-length filter** (`vidaudit/data/filters.py`): measures duration→label leakage and enforces a common frame budget so length cannot stand in for content.

`fetch-data` runs both on your downloaded clips and writes a ready-to-extract manifest + `provenance.json`; `vidaudit/data/croissant.py` emits ML Croissant metadata for the released feature tables.
```bash
python run.py fetch-data --manifest provenance.csv --sources ~/genvidbench --out data/gvb --min-frames 16
python run.py extract temporalspec --manifest data/gvb/manifest.csv --out features/ts.csv
```
**What gets released.** Not the videos — only the *derived* artifacts, which are redistributable: the per-clip **feature tables**, the **LOGO/RvR splits**, the **provenance labels**, the **canonical recipe**, and the **model-weight mirror**. These are a few GB (too large for git), so they will be published to a public mirror (host TBD); you reconstruct the clips themselves locally with `fetch-data`.

## Training
Training is a first-class, standardized subsystem, not an optional hook. The trainer learns a classifier head over a precomputed feature table (the `extract` output), so it reuses the extract pipeline and never re-decodes video in the loop; preprocessing (median-impute → z-score) matches the audit and is persisted in the checkpoint. Defaults live in `scripts/train/<model>.sh`; loss / optimizer / scheduler / head are name-addressable registries you extend with a one-line decorator, and every knob is overridable on the command line (repeatable `--set key=value`, later wins; unknown keys route to `cfg.extra`):
```bash
# extract once, then train a head over the table
python run.py extract temporalspec --manifest clips.csv --out features/train.csv
scripts/train/mlp-probe.sh features/train.csv          # documented defaults

# override loss / lr / schedule / head with no code edits
OUT=runs/probe scripts/train/mlp-probe.sh features/train.csv \
  --set loss=focal --set focal_gamma=1.5 --set lr=3e-4 --set head=linear --set epochs=100
```
`MLP-Probe` (the trainable counterpart to the audit's fixed L2-LR readout) ships as the reference that validates the trainer end-to-end; paper-specific trainable heads (ReStraV, NSG-VD) plug in via the same `build_model` + `default_train_config`. For the paper we run **eval only** (published weights and native heads through the audit); the trainer ships and is validated, but we do not retrain methods to manufacture results.

## Roadmap
- [x] Plugin API (eval / load-weights / train as three orthogonal capabilities)
- [x] Unified conda environment (pinned + lockfile)
- [x] Audit engine in `vidaudit/audit/` (matched-cell LOGO + RvR + the metric tuple + both verdicts), reproduces the paper numbers on real feature CSVs
- [x] `run.py` CLI: `extract`, `eval --features`, `eval --scores`, `train`, `leaderboard`, `fetch-weights`, `fetch-data`
- [x] Closed extract/train → eval loop: `audit_scores` audits a native or trained head's own scores (no readout) through the same LOGO + RvR + metric tuple + verdicts
- [x] Detector wrappers for all 8 baselines (auto-download tier + checkpoint tier), each smoke-tested from clips
- [x] Weight-fetch zoo (`fetch_weights` + `zoo/manifest.yaml`, sha256-verified)
- [x] Standardized data package: P1 canonical re-encode + P2 length filter + local reconstruct (`fetch-data`) + Croissant emitter
- [ ] Public release: feature tables + weight mirror + a hosted leaderboard (host TBD) `[planned]`
- [x] Standardized trainer (`TrainConfig` + registries + uniform loop), validated end-to-end on `MLP-Probe`
- [x] First per-detector recipe: ReStraV trainable head over the standard trainer (`scripts/train/restrav.sh`)
- [ ] WaveRep / NSG-VD trainable heads `[next]`
- [ ] AIGVDBench combined cross-dataset cell (D3 re-run at XCLIP-B/16) `[planned]`

See `FRAMEWORK.md` for the full design, the plugin contract, and the build phases.

## Citation
_To be added after publication._

## License
MIT (see `LICENSE`). Wrapped detectors and datasets retain their original licenses; see each entry in the model zoo and data package.

---
<sub>Keywords: AI-generated video detection, synthetic video detection, deepfake video detection, video forensics, generative video benchmark, detection leaderboard, evaluation toolkit, model zoo, GenVidBench, AIGVDBench, AIGC video, diffusion video detection.</sub>
