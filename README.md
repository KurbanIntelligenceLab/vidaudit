<h1 align="center">VidAudit</h1>
<p align="center"><b>An audited benchmark, leaderboard, and toolkit for AI-generated video detection.</b></p>

VidAudit is a standardized, **audited** evaluation suite for **AI-generated, synthetic, and deepfake video detection**. Clone it to evaluate any detector under one rigorous protocol, retrain detectors that ship training code, or drop in your own method behind a small plugin API and land on a shared, audited leaderboard.

It exists because every AI-video-detection paper currently evaluates differently, a 20-paper survey shows none apply all six standard controls, and high leaderboard AUCs do not predict deployable recall. VidAudit makes the audited protocol the default and makes fair, reproducible comparison a single command.

## What's inside
- **Six-control audit protocol (P1–P6)** — canonical re-encode, clip-length leakage audit, real-vs-real dataset-identity floor, matched-harness re-training, multi-seed/bootstrap CIs, and a true cross-dataset cell.
- **Audited leaderboard** — every detector scored on a matched cell and labeled *certified / caught (dataset identity) / collapses (operating point) / leakage*, with the full audited metric tuple (AUC, above-floor margin, TPR@FPR=1%/0.1%, calibration), not just AUC.
- **Model zoo** — published-weight detectors wrapped behind one plugin API.
- **Standardized data package** — per-clip features + LOGO splits + provenance + Croissant metadata, combining GenVidBench and AIGVDBench (bring-your-own-videos; we don't redistribute source clips).
- **Unified eval + training scripts** — `python run.py eval --model X` and `python run.py train --model X`.

## Quickstart
```bash
pip install -e .
python run.py eval --model temporalspec --cell genvidbench27k   # audit one detector
python run.py leaderboard                                       # (re)render the audited board
```

## Add your method (plugin API)
Subclass `Detector`, implement the native head (and optionally features/training), and register it. The audit, metrics, and leaderboard row come for free:
```python
from vidaudit.detectors import Detector, DetectorSpec, register

@register("mymethod")
class MyMethod(Detector):
    spec = DetectorSpec(name="MyMethod", published_weights=True, training_code=False,
                        backbone="ViT-L/14", family="frame-pixel", paper="arXiv:25xx")
    def score(self, clip):            # native head -> p(generated)
        return my_model(clip.path)
# python run.py eval --model mymethod   ->   P1-P6 audit + a leaderboard row
```

## Model zoo (seed roster)
| detector | backbone | weights | role |
|---|---|---|---|
| TemporalSpec | codec motion vectors | ours | white-box control |
| D3 | XCLIP-ViT-B/16 | training-free | published detector |
| ReStraV | DINOv2 ViT-S/14 | public | published detector |
| WaveRep | DINOv2 ViT-B/14 + wavelet aug | public | published detector |
| NSG-VD | per repo | public | published detector |
| FVMD / RAFT / CLIP | point-tracker / optical flow / CLIP-B/32 | repurposed | reference baselines |

See `FRAMEWORK.md` for the full design, the plugin contract, and the build roadmap.

## Data
We **cannot redistribute** the GenVidBench / AIGVDBench source videos (copyright). VidAudit ships the standardized **features, splits, provenance labels, and the canonical-pipeline recipe**; `vidaudit/data/fetch.py` reconstructs clips from the original sources on your machine.

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
MIT (see `LICENSE`).

---
<sub>Keywords: AI-generated video detection, synthetic video detection, deepfake video detection, video forensics, generative video benchmark, detection leaderboard, evaluation toolkit, model zoo, GenVidBench, AIGVDBench.</sub>
