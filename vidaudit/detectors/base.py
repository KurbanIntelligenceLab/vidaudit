"""Plugin API for the audited AI-generated-video detection toolkit.

A *detector* is anything that scores a video clip as real-vs-generated. To add a
new method you subclass `Detector`, fill in a `DetectorSpec`, implement at least
one of `score()` / `features()`, optionally implement `train()`, and decorate the
class with `@register("name")` (see registry.py). The audit harness then runs it
through the six controls (P1-P6) and the audited metric tuple unchanged.

Design notes
------------
* `score()` is the *native head*: the published per-clip decision the leaderboard
  reports. Return p(generated) in [0, 1] if you can; any monotone score also works
  (the metrics are threshold/rank based).
* `features()` is optional: a per-clip vector fed to the uniform matched-harness
  readout (median-impute -> z-score -> L2-LR, leave-one-generator-out). This is how
  we compare *representations* on equal footing across detectors.
* `train()` is optional: implement it only for methods whose authors released a
  training recipe. Evaluation never requires it.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class Clip:
    """One video clip after the canonical pipeline (P1 re-encode + P2 K-filter)."""
    video_id: str
    path: str                     # local path to the canonical-re-encoded mp4
    source: str                   # generator name, or a real-source name (e.g. "vript")
    is_real: int                  # 1 real, 0 generated
    mv_path: Optional[str] = None   # cached codec motion-vector dump, if extracted
    n_noni_frames: Optional[int] = None
    meta: dict = field(default_factory=dict)


@dataclass
class DetectorSpec:
    """Static metadata for a detector (drives the zoo table + leaderboard columns)."""
    name: str
    published_weights: bool          # are pretrained weights publicly released?
    training_code: bool              # did the authors release training code?
    backbone: str = ""               # e.g. "DINOv2 ViT-B/14", "XCLIP-ViT-B/16", "codec MV"
    family: str = ""                 # "motion" | "frame-pixel" | "frequency" | "trajectory" | ...
    cost_ms: Optional[float] = None  # per-clip POST-DECODE inference cost (ms); None = estimate
    needs_gpu: bool = False
    weights_url: str = ""
    paper: str = ""                  # arXiv id / venue, for the bib + zoo table
    notes: str = ""


class Detector(ABC):
    """Subclass + @register to add a method to the zoo and leaderboard.

    Implement at least one of score() / features(). Implement train() only if the
    method is retrainable through this toolkit.
    """

    spec: DetectorSpec  # set on the subclass (or in __init__)

    # ---- evidence interfaces (override at least one) -------------------------
    def score(self, clip: Clip) -> float:
        """Native head: per-clip p(generated) (or any monotone score)."""
        raise NotImplementedError

    def features(self, clip: Clip) -> np.ndarray:
        """Optional per-clip feature vector for the matched-harness L2-LR readout."""
        raise NotImplementedError

    # ---- optional standardized training -------------------------------------
    def train(self, train_clips: Sequence[Clip], out_dir: str, **kw) -> str:
        """Standardized retraining hook. Return the path to the produced weights.

        Default: this detector is evaluated from published weights and exposes no
        in-toolkit training wrapper.
        """
        raise NotImplementedError(
            f"{self.spec.name}: no standardized training wrapper "
            f"(evaluate published weights, or contribute a train() wrapper)."
        )

    # ---- introspection (used by the harness + zoo table) --------------------
    @property
    def has_native_head(self) -> bool:
        return type(self).score is not Detector.score

    @property
    def has_features(self) -> bool:
        return type(self).features is not Detector.features

    @property
    def is_trainable(self) -> bool:
        return type(self).train is not Detector.train

    def __repr__(self) -> str:  # pragma: no cover
        s = getattr(self, "spec", None)
        return f"<Detector {getattr(s, 'name', type(self).__name__)}>"
