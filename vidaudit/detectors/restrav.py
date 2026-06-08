"""ReStraV (Internò et al., NeurIPS 2025, arXiv:2507.00583): perceptual-straightening
features. Sample 24 frames over a 2-second window centred on the clip, embed each with
a frozen DINOv2 ViT-S/14 (CLS + patch tokens, flattened), then summarise the geometry of
the resulting trajectory Z[T, D]:

    d[t]     = ||Z[t+1] - Z[t]||                 (step distance)
    theta[t] = angle(dZ[t], dZ[t+1]) in degrees  (turning angle)

`features(clip)` returns the 21-d vector [d[0:7], theta[0:6], moment4(d), moment4(theta)]
(ported verbatim from the cluster run_restrav_features.py). DINOv2-S auto-downloads via
torch.hub, so no manual setup is needed.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_NUM_FRAMES = 24
_WINDOW_SEC = 2.0
_IMG = 224


@register("restrav")
class ReStraV(Detector):
    spec = DetectorSpec(
        name="ReStraV", backbone="DINOv2 ViT-S/14", family="appearance",
        published_weights=False, trainable=True, needs_gpu=True,
        paper="Internò et al., NeurIPS 2025 (arXiv:2507.00583)",
        notes="21-d perceptual-straightening trajectory geometry over frozen DINOv2-S "
              "(torch.hub auto-download); no external weights. Trainable: the standard "
              "trainer fits a head over the 21-d features (the published features->audit "
              "L2-LR readout remains the leaderboard row).",
    )

    def __init__(self, n_frames: int = _NUM_FRAMES, window_sec: float = _WINDOW_SEC,
                 size: int = _IMG, device=None):
        self.n_frames = n_frames
        self.window_sec = window_sec
        self.size = size
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            self._device = self._device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
            model = model.eval().to(self._device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        return self._model

    @staticmethod
    def _moment4(x):
        import torch
        return torch.stack([x.mean(dim=-1), x.amin(dim=-1), x.amax(dim=-1),
                            x.var(dim=-1, unbiased=False)], dim=1)

    def _features_from_Z(self, Z):
        """ReStraV temporal geometry on Z[B, T, D] -> [B, 21]."""
        import torch
        import torch.nn.functional as F
        delta = Z[:, 1:, :] - Z[:, :-1, :]                          # [B, T-1, D]
        d = delta.norm(dim=-1)                                      # [B, T-1]
        cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)  # [B, T-2]
        theta = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))     # [B, T-2]
        # Degenerate guard: when either adjacent step vector is zero-norm (a static or
        # duplicated-frame trajectory) the turning angle is geometrically undefined.
        # F.cosine_similarity's eps guard silently returns 0.0 there, which acos maps to a
        # spurious 90deg "turn" for a path that never moves. A non-moving transition has no
        # turn, so we define theta=0 (no turning) wherever a step has no length, rather than
        # emitting a confident right angle. Non-degenerate transitions (both steps nonzero)
        # are untouched, so all existing outputs are preserved exactly.
        zero_step = d <= 0.0                                        # [B, T-1]
        undefined = zero_step[:, :-1] | zero_step[:, 1:]            # [B, T-2]
        theta = torch.where(undefined, torch.zeros_like(theta), theta)
        stats = torch.cat([self._moment4(d), self._moment4(theta)], dim=1)
        return torch.cat([d[:, :7], theta[:, :6], stats], dim=1)    # [B, 21]

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        from vidaudit.detectors._extract import decode_clip
        model = self._load()
        frames = decode_clip(clip.path, n_frames=self.n_frames, size=self.size,
                             window_sec=self.window_sec)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(self._device) / 255.0
        with torch.no_grad():
            feats = model.forward_features(x)
            tokens = torch.cat([feats["x_norm_clstoken"].unsqueeze(1),
                                feats["x_norm_patchtokens"]], dim=1)   # [T, 1+P, D]
            Z = tokens.flatten(1).unsqueeze(0)                         # [1, T, (1+P)*D]
            vec = self._features_from_Z(Z)[0]
        return vec.detach().cpu().numpy()

    # ---- trainable head over the 21-d features (the standard trainer drives this) ----
    def default_train_config(self):
        """A small head suits the 21-d feature vector; train more epochs since it is light."""
        from vidaudit.train.config import TrainConfig
        return TrainConfig(head="mlp", hidden=(64,), dropout=0.1, loss="bce",
                           optimizer="adamw", lr=1e-3, weight_decay=1e-4,
                           scheduler="cosine", epochs=80, batch_size=256)

    def build_model(self, cfg):
        from vidaudit.train.registries import build_head
        in_dim = cfg.extra.get("in_dim")
        if not in_dim:
            raise ValueError("the trainer sets cfg.extra['in_dim'] from the feature table "
                             "before build_model(); extract ReStraV features first")
        return build_head(cfg.head, in_dim, cfg)
