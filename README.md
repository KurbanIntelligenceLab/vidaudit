<h1 align="center">VidAudit</h1>

<p align="center">
  <b>An audited benchmark, leaderboard, and toolkit for AI-generated video detection.</b><br>
  <sub>Evaluate any detector under one rigorous protocol. Train any method through a uniform script. Add your own behind a small plugin API.</sub>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-blue.svg">
  <img alt="Audit protocol" src="https://img.shields.io/badge/audit-P1--P6-8A2BE2.svg">
  <a href="#"><img alt="HuggingFace dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-coming-lightgrey.svg"></a>
  <a href="#"><img alt="Leaderboard Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20leaderboard-coming-lightgrey.svg"></a>
  <a href="#"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-coming-b31b1b.svg"></a>
</p>

VidAudit is a standardized, **audited** evaluation suite for **AI-generated, synthetic, and deepfake video detection**. It exists because every AI-video-detection paper currently evaluates differently, a 20-paper survey shows none apply all six standard controls, and high leaderboard AUCs do not predict deployable recall. VidAudit makes the audited protocol the default and makes fair, reproducible comparison (and honest re-ranking by deployability) a single command.

---

## 🏆 Audited leaderboard

Every detector is scored on the **matched 27k-clip GenVidBench cell** under leave-one-generator-out (LOGO) evaluation. The headline number papers report is **LOGO-OOD AUC**. The audit adds three columns that decide whether that number is real: the **real-vs-real floor** (`RvR`), the **above-floor margin**, and **deployable recall** (`TPR@0.1%`). Sorted by OOD AUC, so you can watch high-AUC methods separate from genuinely robust ones.

| # | Model | LOGO-OOD AUC ↑ | RvR floor | Margin ↑ | TPR@0.1% ↑ | Verdict |
|--:|---|:--:|:--:|:--:|:--:|---|
| 1 | **WaveRep** | 0.996 | 0.534 | +0.462 | **0.816** | ✅ Certified, usable |
| 2 | **XSFF** (ours) | 0.946 | 0.604 | +0.342 | _pending_ | ✅ Certified |
| 3 | **ReStraV** | 0.931 | 0.586 | +0.345 | 0.634 | ✅ Certified, usable |
| 4 | **D3** | 0.887 | 0.421 | +0.466 | _pending_ | ✅ Certified (native head) |
| 5 | **FVMD** | 0.880 | 0.574 | +0.306 | 0.027 | ⚠️ Certified, collapses @0.1% |
| 6 | **TemporalSpec+aug** (ours) | 0.871 | 0.634 | +0.237 | _pending_ | ✅ Certified |
| 7 | **RAFT** | 0.855 | 0.627 | +0.228 | 0.020 | ⚠️ Certified, collapses @0.1% |
| 8 | **CLIP** | 0.852 | 0.766 | +0.086 | 0.238 | ❌ Caught (dataset identity) |
| 9 | **TemporalSpec** (ours) | 0.832 | 0.643 | +0.189 | 0.024 | ⚠️ Certified, collapses @0.1% |
| 10 | **NSG-VD** | 0.660 | 0.596 | +0.064 | _pending_ | ❌ Near-floor (rides identity) |

> **Existence proof.** A trivial 3-feature **clip-length probe** scores **0.998** AUC under an unaudited protocol and **0.529** after the P2 leakage filter. That gap is why the audit exists: a near-perfect leaderboard score can be almost entirely confound. See the paper for the full derivation.

**How to read the verdict.**
- ✅ **Certified**: clears its real-vs-real floor by a wide margin, genuine cross-generator signal.
- ✅ **Certified, usable**: also keeps useful recall at a deployable false-positive rate.
- ⚠️ **Certified, collapses @0.1%**: strong AUC, but recall at FPR = 0.1% falls to near zero. The ranking and the deployability disagree.
- ❌ **Caught (dataset identity)**: a high real-vs-real AUC means the score largely reflects *which dataset the reals came from*, not generation artifacts.
- ❌ **Near-floor**: barely exceeds the dataset-identity floor at all.

**Column definitions.**
- **LOGO-OOD AUC**: leave-one-generator-out AUC on held-out generators.
- **RvR floor**: real-vs-real AUC (separating two *real* datasets). An artifact-based detector should sit near 0.5 here; a high value means dataset-identity leakage.
- **Margin**: LOGO-OOD minus RvR, the real generalization remaining above the floor.
- **TPR@0.1%**: true-positive rate at a 0.1% false-positive operating point (deployable recall).

<sub>Numbers from the WACV 2027 audit (matched 27k GenVidBench cell, native head or uniform L2-LR readout per method). Bootstrap CIs, the full 116k cell, and the combined GenVidBench + AIGVDBench cross-dataset cell are reported in the paper and will land here as the data package ships. `_pending_` operating points are computed but not yet folded into this table.</sub>

---

## What's inside
- **Six-control audit protocol (P1-P6)**: canonical re-encode, clip-length leakage filter, real-vs-real dataset-identity floor, matched-harness re-training, multi-seed/bootstrap CIs, and a true cross-dataset cell.
- **Audited leaderboard**: every detector labeled by the two verdicts above, with the full metric tuple (AUC, above-floor margin, TPR@FPR, calibration), not just AUC.
- **Model zoo**: 8 detectors behind one plugin API; backbones auto-download or load published checkpoints (sha256-verified via the zoo).
- **Standardized data package**: per-clip features, LOGO splits, provenance, and Croissant metadata, combining GenVidBench and AIGVDBench (bring-your-own-videos; we do not redistribute source clips).
- **Unified CLI**: `run.py extract` (clips → features) → `run.py eval` (audit → verdicts) → `run.py leaderboard`, plus a uniform, overridable trainer driven by shell scripts.

## Installation
```bash
git clone <repo-url> vidaudit && cd vidaudit
conda env create -f environment.yml     # one unified env for the whole repo
conda activate vidaudit
pip install -e .                         # register vidaudit (editable); runtime deps come from conda
```
One environment covers everything (audit, figures, video/codec tooling, deep backbones, training). Apple Silicon (MPS) and Linux/CUDA are both supported; heavy GPU runs are meant for a cluster, the same spec works locally for development. Dependencies are pinned to ranges in `environment.yml`; for a byte-exact environment, use `conda env create -f environment.lock.yml` instead.

## Usage
```bash
# 1. Extract a detector's per-clip features from your clips -> a CSV
python run.py extract restrav --manifest clips.csv --out restrav.csv
python run.py extract waverep --manifest clips.csv --out waverep.csv \
       --weights /path/to/weights_dinov2_G4.ckpt        # weighted detectors take a checkpoint

# 2. Audit the feature table -> the full metric tuple + both verdicts (one leaderboard record)
python run.py eval --features restrav.csv --subset baseline_clip_subset.csv

# 3. (Re)render the audited leaderboard
python run.py leaderboard
```
`clips.csv` is a manifest with columns `(video_id, generator, label, is_real, mp4_path)`. The
auto-download detectors (D3, ReStraV, CLIP, RAFT) need no setup; TemporalSpec needs only the
bundled PyAV/ffmpeg; WaveRep/FVMD/NSG-VD take a checkpoint (`--weights`, or fetched from the zoo).
`run.py train <model> --features <table>` learns a head over the extracted table (see [Training](#training)); `fetch-data` is planned (see the roadmap).

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

Vendored model code (PIPs++ for FVMD, the NSG-VD codebase) lives under `vidaudit/_vendor/`, kept separate from our thin wrappers and carrying its upstream license. Published checkpoints (WaveRep G4, NSG-VD per-generator + the ADM diffusion model) are currently on the project's permanent cluster storage with their sha256s recorded in `zoo/manifest.yaml`; a public mirror is planned.

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
- 🤗 **HuggingFace dataset / leaderboard Space**: _(coming)_

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
- [x] `run.py` CLI: `extract`, `eval --features`, `train`, `leaderboard`, `fetch-weights`, `fetch-data`
- [x] Detector wrappers for all 8 baselines (auto-download tier + checkpoint tier), each smoke-tested from clips
- [x] Weight-fetch zoo (`fetch_weights` + `zoo/manifest.yaml`, sha256-verified)
- [x] Standardized data package: P1 canonical re-encode + P2 length filter + local reconstruct (`fetch-data`) + Croissant emitter
- [ ] Public weight mirror (host the cluster checkpoints over HTTP) `[planned]`
- [ ] HuggingFace dataset + leaderboard Space `[planned]`
- [x] Standardized trainer (`TrainConfig` + registries + uniform loop), validated end-to-end on `MLP-Probe`
- [ ] Per-detector paper training recipes (ReStraV / WaveRep / NSG-VD heads) `[next]`
- [ ] AIGVDBench combined cross-dataset cell (D3 re-run at XCLIP-B/16) `[planned]`

See `FRAMEWORK.md` for the full design, the plugin contract, and the build phases.

## Citation
```bibtex
@inproceedings{vidaudit,
  title  = {How Inflated Are AI-Generated Video Detection Benchmarks?},
  author = {Anonymous},
  year   = {2027},
  note   = {VidAudit toolkit and audited leaderboard}
}
```

## License
MIT (see `LICENSE`). Wrapped detectors and datasets retain their original licenses; see each entry in the model zoo and data package.

---
<sub>Keywords: AI-generated video detection, synthetic video detection, deepfake video detection, video forensics, generative video benchmark, detection leaderboard, evaluation toolkit, model zoo, GenVidBench, AIGVDBench, AIGC video, diffusion video detection.</sub>
