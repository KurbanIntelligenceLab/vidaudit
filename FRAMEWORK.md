# vidaudit : audited evaluation + training toolkit and leaderboard for AI-generated video detection

> Working name; rebrand TBD (you + advisor).

## Goal
One repo a newcomer can clone to:
1. **Evaluate** any detector through a standardized **audited** protocol (P1-P6) and the audited metric tuple, with one command (published weights or training-free native heads; no training required).
2. **Train / retrain** any method that ships a recipe through a uniform, fully-overridable trainer driven by a shell script (defaults in the script, overrides on the command line).
3. **Extend** the platform: add a method behind a small plugin API, or add a dataset behind a small adapter, and land on the shared audited leaderboard.

Backed by a standardized **data package** (features + splits + provenance; bring-your-own-videos) and a **model zoo** of published-weight baselines with verified download links.

## Why
Every AIGV-detection paper currently evaluates differently; the survey shows none apply all six controls; and high leaderboard AUCs do not predict deployable recall. This toolkit makes the audited protocol the default, makes fair reproducible comparison (and honest re-ranking by deployability) a single command, and gives future work one place to train, evaluate, and compare on equal footing.

## Three orthogonal capabilities (the detector contract)
A detector (`vidaudit/detectors/base.py`) is anything that scores a clip real-vs-generated. It can do up to three independent things:

| Capability | Interface | Who has it |
|---|---|---|
| **Evaluate** | `score()` (native head) and/or `features()` (uniform L2-LR readout) | every detector |
| **Load weights** (eval-only) | `load_weights(path=None)` : auto-fetch + sha256-verify from the zoo | pretrained methods |
| **Train** | `build_model(cfg)` + `default_train_config()`, driven by the standard trainer | methods with `trainable=True` |

* `score(clip)` is the method's own published decision (the leaderboard's "native" row). `has_native_head` reports whether it is implemented.
* `features(clip)` is a per-clip vector fed to the *uniform* readout (median-impute, z-score, L2-LR, leave-one-generator-out), comparing representations on equal footing. `has_features` reports whether it is implemented.
* A detector may expose either or both. WaveRep has both (native LLR 0.996 plus a raw-768d readout that rides dataset identity); RAFT has only features; a black-box API detector may have only a score.

**Training-free is a method property, not a platform gap.** D3 (training-free) and frozen references (CLIP, RAFT, FVMD) declare `trainable=False` and are eval-only by design. Every method that *does* have a recipe is trainable through the same uniform path; training is a first-class pillar, not an optional afterthought.

## Plugin contract (add a method)
Subclass `Detector`, set a `DetectorSpec`, and:
- implement **at least one** of `score()` / `features()` (evaluation), and
- if pretrained weights exist, set `weights_url` (+ `weights_sha256`) and implement `load_weights()`, and
- if the method is trainable, set `trainable=True` and implement `build_model(cfg)` + `default_train_config()`.

Decorate with `@register("name")`. The audit harness then handles P1-P6, the metrics, the verdict, and the leaderboard row. No training loop to write (the standard trainer calls `build_model`); override `train()` only for a non-standard loop.

## Training subsystem (first-class)
A single serializable `TrainConfig` (`vidaudit/train/config.py`, torch-free) is the hyperparameter surface; loss / optimizer / scheduler / head are **name-addressable through registries** (`vidaudit/train/registries.py`), so a user adds their own with a one-line decorator and references it by name without touching framework code.

```python
@dataclass(slots=True)
class TrainConfig:
    # data: a precomputed per-clip feature table (the `extract` output)
    features=""; val_features=""; feature_cols=None; subset=None; val_frac=0.2
    # head (registry: linear | mlp)
    head="mlp"; hidden=(256,); dropout=0.1
    # optimization (each name resolves against a registry)
    loss="bce"; optimizer="adamw"; lr=1e-3; weight_decay=1e-4; momentum=0.9
    scheduler="cosine"; epochs=50; batch_size=256; amp=True; grad_clip=1.0
    seed=42; device="auto"; extra={}    # unknown --set keys land in extra
```

Registries: `@register_loss` (`bce`, `focal`), `@register_optimizer` (`adamw`, `adam`, `sgd`), `@register_scheduler` (`cosine`, `step`, `none`), `@register_head` (`linear`, `mlp`). The standard `SupervisedTrainer(detector, cfg)` (`vidaudit/train/trainer.py`) owns the loop — seeding, AMP, grad-clip, per-epoch validation AUC, best-checkpoint selection — and calls `detector.build_model(cfg)`; a method needing a bespoke loop overrides `Detector.train()`. The trainer learns a head over a **precomputed feature table** (the same CSV the audit consumes), so it reuses the extract pipeline and never re-decodes video in the loop; preprocessing (median-impute → z-score) matches the audit and is persisted in the checkpoint. Label is generated-positive (`y = 1 - is_real`).

**Shell scripts hold the defaults; the command line holds the freedom.**
```bash
# extract once, then train a head over the table:
python run.py extract temporalspec --manifest clips.csv --out features/train.csv
scripts/train/mlp-probe.sh features/train.csv            # documented defaults live in the script

# user overrides, no code edits (repeatable --set, later wins; unknown keys -> cfg.extra):
OUT=runs/probe scripts/train/mlp-probe.sh features/train.csv \
  --set loss=focal --set focal_gamma=1.5 --set lr=3e-4 --set head=linear --set epochs=100
```
Precedence: detector `default_train_config()` < the recipe script's `--set` flags < user `--set` flags. The checkpoint (`<out>/model.pt`) carries the state dict + config + feature columns + preprocessing stats + best val AUC; `<out>/metrics.json` holds the per-epoch history. The **official** numbers come from the audit (`run.py eval`) — the trainer's val AUC only selects the checkpoint.

`MLP-Probe` (the standardized trainable readout, the trainable counterpart to the audit's fixed L2-LR) ships as the reference that validates this subsystem end-to-end. A paper-specific trainable detector (a ReStraV MLP head over DINOv2 features, the NSG-VD discriminator) plugs in identically: implement `build_model(cfg)` + `default_train_config()` and the same trainer drives it.

## Model zoo + weights (download, verify, cache)
We host or link published weights with a verifiable manifest. `zoo/manifest.yaml`:
```yaml
waverep:
  weights: [{ url: "https://.../waverep_dinov2b14.pth", sha256: "...", size_mb: 350 }]
  license: "NVIDIA non-commercial"
  backbone: dinov2_vitb14
```
`run.py fetch-weights <model>` downloads into `~/.cache/vidaudit/weights/<name>/` and verifies the hash; `run.py eval <model>` auto-fetches if missing, or takes `--weights <path>`. Hosting uses the advisor's funded storage where authors did not release a stable link. Same fetch/verify/cache code path as the data package (`data/fetch.py`).

## Data package + dataset extensibility
We **cannot redistribute** GenVidBench / AIGVDBench videos (copyright). The released "dataset" is: per-clip **features** per wrapped detector, the canonical-re-encode + K-filter **recipe**, matched-cell **membership + LOGO splits + provenance labels**, and **Croissant** metadata. `data/fetch.py` reconstructs clips from the original sources on the user's machine.

**Adding a future dataset is a thin adapter.** Datasets live behind a registry, so the audit pipeline never changes:
```python
@register_dataset("genvidbench")
class GenVidBench(VideoDataset):
    spec = DatasetSpec(
        name="genvidbench",
        generators=[...],
        real_sources=["hd_vg_130m", "vript"],  # len>=2 enables P3 RvR; len<2 auto-disables it
        fetch=<recipe>,                          # reconstruct clips locally (no redistribution)
    )
    def clips(self, split, cell): ...            # yields Clip objects with provenance
```
The audit (P1-P6, matched cells, LOGO) consumes the abstract `Clip` / feature stream, so a new dataset gets the entire pipeline, metrics, verdict, and a leaderboard section for free. `--dataset a+b` builds a **combined** cell with source tags (the GenVidBench+AIGVDBench cell). RvR (P3) auto-disables when a dataset ships fewer than two real sources (handles AIGVDBench). To add a dataset you write only the adapter (enumerate clips, label generators and real sources, point at a download recipe) plus its manifest.

## Audit (dataset- and detector-agnostic)
`audit/protocol.py` orchestrates the six controls into one record per detector; `audit/metrics.py` computes AUC, pAUC@10%, TPR@1%, TPR@0.1%, Brier, ECE, RvR, margin, and bootstrap CIs; `audit/verdict.py` assigns the two verdicts below; `audit/leaderboard.py` writes the rows. Because these operate on the abstract evidence stream, every new detector or dataset is covered automatically.

## Leaderboard schema
Two **orthogonal** verdicts, mirroring the two paper figures:
- **floor_verdict** (does LOGO-OOD clear the method's own real-vs-real floor?): `certified` | `caught` (high RvR, rides dataset identity) | `marginal` (small margin) | `leakage` (relies on a removed confound).
- **deploy_tier** (recall at a deployable FPR, from TPR@0.1%): `usable` | `marginal` | `collapses`.

Columns: `model, ours, family, backbone, weights, benchmark, cell, readout, LOGO-ID, LOGO-OOD, RvR, margin, pAUC@10%, TPR@1%, TPR@0.1%, Brier, ECE, cost_ms, floor_verdict, deploy_tier, source, paper, notes`.

## Repo layout
```
vidaudit/
  vidaudit/
    detectors/   base.py (contract)  registry.py  _extract.py (clips->table)
                 temporalspec.py d3.py restrav.py waverep.py nsgvd.py fvmd.py
                 raft.py clip.py  probe.py (trainable readout)
    data/        datasets/base.py (Dataset ABC + registry)  datasets/<name>.py
                 cells.py (matched/LOGO)  canonical.py (P1) filters.py (P2) fetch.py [planned]
    audit/       protocol.py (P1-P6)  metrics.py  verdict.py  leaderboard.py
    train/       config.py (TrainConfig)  trainer.py (SupervisedTrainer)
                 registries.py (loss/optim/sched/head)  data.py (feature tables)
    features/    mv.py (shared 13-d codec/MV extractor)
  scripts/       train/<model>.sh + README   (defaults + repeatable --set passthrough)
  zoo/           manifest.yaml (weights: url + sha256 + license)
  run.py         extract | eval | train | leaderboard | fetch-weights[planned] | fetch-data[planned]
  leaderboard.csv  LEADERBOARD.md  README.md  FRAMEWORK.md
```

## Python and environment
The reference environment (`environment.yml`) pins **Python 3.14**, the newest interpreter with reliable prebuilt wheels across the stack (numpy, scipy, PyAV, opencv, torch, tokenizers, pyarrow), verified all-binary with no source-build fallbacks. **Library versions are pinned with the compatible-release operator** (`~=X.Y.Z`, patch updates only, minor/major locked): modern baselines that block breaking minor and major releases, so `conda env create` is reproducible and upstream cannot silently break it. `environment.lock.yml` (committed) gives byte-exact recreation, and pins are bumped deliberately when modernizing. torch/torchvision come from **conda-forge, not pip**: pip's torch bundles its own OpenMP runtime that aborts with `OMP Error #15` against the conda scientific stack on macOS, so conda-forge (one shared OpenMP runtime) is the reliable choice even though it trails PyPI by a minor version. Use current library APIs freely.

The **package stays portable to the `requires-python` floor (3.10+)**: we use portable modern features where useful (e.g. `@dataclass(slots=True)`) but avoid anything that would raise the floor (PEP 695 generics, `match`-only dispatch, `typing.override`, relying on PEP 649 lazy annotations). `from __future__ import annotations` stays for back-compat (still valid on 3.14). To adopt a floor-raising language feature, bump `requires-python` and `environment.yml` together; do not raise the floor silently.

## Model zoo (seed roster)
| detector | backbone | weights | trainable | role |
|---|---|---|---|---|
| TemporalSpec (ours) | codec motion vectors (13-d) | ours | yes | white-box control |
| D3 | XCLIP-ViT-B/16 | training-free | no | published detector |
| ReStraV | DINOv2 ViT-S/14 | public | yes | published detector |
| WaveRep | DINOv2 ViT-B/14 + wavelet aug | public | partial | published detector |
| NSG-VD | diffusion noise-score | public (ckpts) | yes | published detector |
| FVMD | PIPs++ point tracker | metric | no | repurposed reference |
| RAFT | RAFT-Large flow | pretrained | no | repurposed reference |
| CLIP | CLIP-ViT-B/32 | pretrained | no | repurposed reference |

Excluded for now (no public weights): **DeMamba** (authors withhold, GitHub issues #5/#16/#21), DeCoF / ATSS / CMTA / VidGuard-R1 (no release); kept on a "wanted: weights" list.

## Scope discipline (for the WACV paper)
- **Eval is what we run for the paper:** published weights, native heads, and training-free scores through the audit. This populates the leaderboard and surfaces failing models (high AUC but collapses at a deployable threshold, or rides dataset identity).
- **Training is built and real, but not exercised for headline numbers:** the standard trainer + shell-script recipes ship and are validated end-to-end on the `MLP-Probe` reference (train → checkpoint → reload → predict), so the platform is genuinely reproducible and extensible. We do not retrain methods to manufacture paper results.

## Build phases (engineering order, not deadline-driven)
1. **Core eval path** : `base.py`, `registry.py`, `run.py`, `audit/{metrics,verdict,leaderboard}.py`, `data/{datasets/base,cells}.py`, reusing the existing matched-cell + LOGO + metric code (`Baselines/evaluate_features.py`, `run_audited_metrics.py`).
2. **Eval wrappers** for the seed roster (reuse `Baselines/run_*_features.py`). Each yields a leaderboard row and surfaces failing models. (D3-native + NSG-VD audits already run on the cluster: first rows in hand.)
3. **Training subsystem** : `train/{config,trainer,losses,optim,augment}.py` + `scripts/train/*.sh` + `build_model` / `default_train_config` for the trainable roster (TemporalSpec, ReStraV, WaveRep, NSG-VD); validated end-to-end on at least one.
4. **Data + weights distribution** : `canonical.py` + `filters.py` + `fetch.py` + the standardized data package + Croissant (HF dataset) + `zoo/manifest.yaml` + weight hosting.
5. **Release** : leaderboard render + HF Space + README + "add your model in ~30 lines" and "add a dataset in ~20 lines" tutorials.

## Status
- [x] Plugin API (eval / load-weights / train as three orthogonal capabilities); training first-class
- [x] Unified conda env (Python 3.14, pinned ranges + lockfile)
- [x] Audit engine (`vidaudit/audit/`): matched-cell LOGO + RvR floor + metric tuple + both verdicts; reproduces the paper numbers on real feature CSVs (ReStraV/TemporalSpec exact, FVMD operating-point exact)
- [x] `run.py` CLI: `extract` (clips → features), `eval --features` (audit), `leaderboard`
- [x] All 8 detector wrappers from clips: D3, ReStraV, CLIP, RAFT, TemporalSpec (clone-and-run, verified) + WaveRep, FVMD (checkpoints, verified) + NSG-VD (ADM diffusion + Swin; verifying). Vendored model code in `vidaudit/_vendor/` (PIPs++, NSG-VD), separate from the thin wrappers
- [x] Weight-fetch zoo (`zoo.py` + `zoo/manifest.yaml`, sha256-verified); checkpoints on cluster permanent storage (public mirror TBD)
- [x] Standardized trainer (`TrainConfig` + loss/optim/sched/head registries + uniform loop), validated end-to-end on `MLP-Probe`
- [ ] Standardized data package + Croissant; HF dataset + Space; per-detector paper training recipes (ReStraV/WaveRep/NSG-VD heads); AIGVDBench combined cell (D3 re-run at XCLIP-B/16)
