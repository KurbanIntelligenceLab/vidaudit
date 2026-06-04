"""MLP-Probe: the framework's standardized *trainable* readout.

This is the trainable counterpart to the audit's fixed L2-logistic-regression
readout. It learns a small MLP (or linear) head over any precomputed per-clip
feature table through the standard trainer, so it doubles as the reference that
validates the training subsystem end-to-end. It is a VidAudit baseline, not a
reproduction of any paper's head.

A paper-specific trainable detector (e.g. a ReStraV MLP head over DINOv2
features, or the NSG-VD discriminator) plugs in identically: implement
`build_model(cfg)` + `default_train_config()` and the same trainer drives it.

Scope: the probe operates on a feature *table* (it owns no video backbone), so it
exposes `predict_table()` rather than `score(clip)`/`features(clip)`. Extract a
backbone's features once, then train/evaluate the probe on that table.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vidaudit.detectors.base import Detector, DetectorSpec
from vidaudit.detectors.registry import register


@register("mlp-probe")
class MLPProbe(Detector):
    spec = DetectorSpec(
        name="MLP-Probe", backbone="features", family="readout",
        published_weights=False, trainable=True, needs_gpu=False,
        notes="standardized trainable head over any feature table; the trainable "
              "counterpart to the audit's L2-LR readout, and the reference that "
              "validates the training subsystem. Not a paper reproduction.",
    )

    def __init__(self):
        self._model = None
        self._ckpt = None

    # ---- training hooks ------------------------------------------------------
    def default_train_config(self):
        from vidaudit.train.config import TrainConfig
        return TrainConfig(head="mlp", hidden=(256, 128), dropout=0.1,
                           loss="bce", optimizer="adamw", lr=1e-3, weight_decay=1e-4,
                           scheduler="cosine", epochs=50, batch_size=256)

    def build_model(self, cfg):
        from vidaudit.train.registries import build_head
        in_dim = cfg.extra.get("in_dim")
        if not in_dim:
            raise ValueError("in_dim missing; the trainer sets cfg.extra['in_dim'] "
                             "from the feature table before build_model()")
        return build_head(cfg.head, in_dim, cfg)

    # ---- use a trained head --------------------------------------------------
    def load_weights(self, path: Optional[str] = None) -> None:
        import torch
        if not path:
            raise ValueError("mlp-probe has no published weights; pass a checkpoint "
                             "produced by `run.py train mlp-probe`.")
        from types import SimpleNamespace

        from vidaudit.train.registries import build_head
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SimpleNamespace(**ckpt["config"])
        model = build_head(cfg.head, len(ckpt["feature_cols"]), cfg)
        model.load_state_dict(ckpt["state_dict"])
        self._model, self._ckpt = model.eval(), ckpt

    def predict_table(self, df) -> np.ndarray:
        """Per-row p(generated) for a feature table, using the loaded head and the
        persisted median-impute + z-score. Feed these scores to the audit, or score
        a held-out table directly."""
        import pandas as pd
        import torch
        if self._model is None:
            raise RuntimeError("call load_weights(<checkpoint>) first")
        feats = self._ckpt["feature_cols"]
        pre = self._ckpt["preproc"]
        X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()
        X = np.where(np.isnan(X), pre["median"], X)
        X = (X - pre["mean"]) / pre["std"]
        with torch.no_grad():
            return torch.sigmoid(self._model(torch.from_numpy(X.astype(np.float32)))).numpy()
