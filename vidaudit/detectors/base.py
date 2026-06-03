"""Plugin API for the audited AI-generated-video detection toolkit.

A *detector* scores a video clip as real-vs-generated. Adding a method means:
subclass `Detector`, fill in a `DetectorSpec`, implement at least one *evidence*
interface (`score()` and/or `features()`), and decorate with `@register("name")`.
If the method ships a training recipe, set `trainable=True` and implement
`build_model()` / `default_train_config()`; the standardized trainer
(`vidaudit.train.trainer`) then trains it through a uniform, fully-overridable
loop. The audit harness runs every detector through the six controls (P1-P6) and
the audited metric tuple unchanged.

Three orthogonal capabilities
-----------------------------
EVALUATE (always available): produce evidence for the audit.
    * `score(clip)`    -> native head: the method's own published per-clip
                          decision. This is the row the leaderboard reports as
                          the method's real performance (WaveRep LLR, D3
                          std-of-2nd-difference, ReStraV MLP head).
    * `features(clip)` -> a per-clip vector fed to the *uniform* matched-harness
                          readout (median-impute -> z-score -> L2-LR,
                          leave-one-generator-out). Compares representations on
                          equal footing, independent of each method's head.
  A detector may expose either or both (WaveRep has both; RAFT has only
  features; a black-box API detector may have only a score).

LOAD WEIGHTS (pretrained methods): `load_weights()` pulls published or trained
  weights (fetched + sha256-verified from the model zoo) so evaluation never
  requires training.

TRAIN (methods with a recipe): declare `trainable=True` and expose
  `build_model` + `default_train_config`; `run.py train <name>` drives the
  standard trainer. Training-free methods (D3) and frozen references (CLIP,
  RAFT, FVMD) stay `trainable=False` and are eval-only by nature, not by
  omission.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:  # keep the contract import-light (no numpy/torch needed to load it)
    import numpy as np
    from vidaudit.train.config import TrainConfig


@dataclass(slots=True)
class Clip:
    """One video clip after the canonical pipeline (P1 re-encode + P2 K-filter)."""
    video_id: str
    path: str                       # local path to the canonical-re-encoded mp4
    source: str                     # generator name, or a real-source name (e.g. "vript")
    is_real: int                    # 1 real, 0 generated
    dataset: str = ""               # owning dataset (e.g. "genvidbench"); set for combined cells
    mv_path: Optional[str] = None   # cached codec motion-vector dump, if extracted
    n_noni_frames: Optional[int] = None
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class DetectorSpec:
    """Static metadata for a detector (drives the zoo table + leaderboard columns)."""
    name: str
    backbone: str = ""               # e.g. "DINOv2 ViT-B/14", "XCLIP-ViT-B/16", "codec MV"
    family: str = ""                 # motion | appearance | forensic | codec | fusion | ...
    # --- capabilities ---
    published_weights: bool = False  # are pretrained weights publicly released?
    trainable: bool = False          # does the method ship a training recipe we wrap?
    needs_gpu: bool = False
    cost_ms: Optional[float] = None  # per-clip POST-DECODE inference cost (ms); None = estimate
    # --- model-zoo weight fetch (mirrors the data package: link + verify + cache) ---
    weights_url: str = ""
    weights_sha256: str = ""
    weights_size_mb: Optional[float] = None
    license: str = ""
    paper: str = ""                  # arXiv id / venue, for the bib + zoo table
    notes: str = ""


class Detector(ABC):
    """Subclass + @register to add a method to the zoo and leaderboard.

    Override at least one evidence interface (score / features). Override the
    training hooks (build_model / default_train_config) only when the method is
    `trainable`. Evaluation never requires training.
    """

    spec: DetectorSpec  # set on the subclass (or in __init__)

    # ---- evidence (override at least one) -----------------------------------
    def score(self, clip: Clip) -> float:
        """Native head: per-clip p(generated) (or any monotone score)."""
        raise NotImplementedError

    def features(self, clip: Clip) -> np.ndarray:
        """Optional per-clip feature vector for the matched-harness L2-LR readout."""
        raise NotImplementedError

    # ---- weights for eval-only use ------------------------------------------
    def load_weights(self, path: Optional[str] = None) -> None:
        """Load published/trained weights for evaluation.

        `path=None` resolves this detector's entry in the model zoo (download +
        sha256-verify into the local cache, like the data package). Training-free
        detectors may leave this unimplemented.
        """
        raise NotImplementedError

    # ---- standardized training (only when spec.trainable) -------------------
    def build_model(self, cfg: "TrainConfig") -> Any:
        """Return the trainable model (e.g. an nn.Module) configured from cfg.

        The standard trainer (vidaudit.train.trainer.SupervisedTrainer) owns the
        loop, optimizer, loss, AMP, checkpointing and eval; this only builds the
        network + head. Override train() instead if the method needs a bespoke
        loop.
        """
        raise NotImplementedError

    def default_train_config(self) -> "TrainConfig":
        """The method's default hyperparameters (what scripts/train/<name>.sh encodes)."""
        raise NotImplementedError

    def train(self, cfg: "TrainConfig", train_clips: Sequence[Clip], out_dir: str) -> str:
        """Train and return the path to the produced weights.

        Default: defer to the standardized trainer, which calls build_model().
        Most trainable detectors implement build_model + default_train_config and
        never touch this; override only for a non-standard loop.
        """
        from vidaudit.train.trainer import SupervisedTrainer  # lazy: torch optional
        return SupervisedTrainer(self, cfg).fit(train_clips, out_dir)

    # ---- introspection (used by the harness + zoo table) --------------------
    @property
    def has_native_head(self) -> bool:
        """True if the detector implements score() (its own published decision)."""
        return type(self).score is not Detector.score

    @property
    def has_features(self) -> bool:
        """True if the detector implements features() (for the uniform L2-LR readout)."""
        return type(self).features is not Detector.features

    @property
    def has_pretrained(self) -> bool:
        """True if published weights are registered for download."""
        return bool(self.spec.weights_url)

    @property
    def is_trainable(self) -> bool:
        """True if the method declares a recipe and exposes build_model()."""
        return bool(self.spec.trainable) and type(self).build_model is not Detector.build_model

    def __repr__(self) -> str:  # pragma: no cover
        s = getattr(self, "spec", None)
        return f"<Detector {getattr(s, 'name', type(self).__name__)}>"
