"""FVMD motion baseline (Liu et al., 2024): a repurposed reference using PIPs++ point
tracking. Ported from the cluster run_fvmd_features.py.

Per clip: decode 16 frames at 256x256, run PIPs++ to track a 20x20 keypoint grid ->
trajectories [1,S,400,2], compute velocity + acceleration, and turn each into the FVMD
HOG-style histogram (calc_hist) -> 512-d each. `features(clip)` = the concatenated 1024-d
[v..., a...] vector. The PIPs++ network is vendored under vidaudit/_vendor/fvmd; its
weights are a public release fetched via the zoo (or pass a local --weights path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_VENDOR = str(Path(__file__).resolve().parents[1] / "_vendor")


def _ensure_vendor():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)


@register("fvmd")
class FVMD(Detector):
    spec = DetectorSpec(
        name="FVMD", backbone="PIPs++ point tracker", family="motion",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="see zoo/manifest.yaml (fvmd): PIPs++ public release",
        paper="FVMD (Liu et al., 2024)",
        notes="PIPs++ trajectories -> velocity/accel HOG histograms (1024-d). Vendored "
              "net in vidaudit/_vendor/fvmd; weights are a public release.",
    )
    feature_names = [f"v{j}" for j in range(512)] + [f"a{j}" for j in range(512)]

    def __init__(self, n_points: int = 400, n_frames: int = 16, size: int = 256,
                 iters: int = 16, weights=None, device=None):
        self.n_points = n_points
        self.n_frames = n_frames
        self.size = size
        self.iters = iters
        self._weights = weights
        self._device = device
        self._model = None

    def load_weights(self, path=None) -> None:
        if path:
            self._weights = path
        self._model = None

    def _resolve_weights(self) -> str:
        if self._weights:
            return str(self._weights)
        from vidaudit.zoo import fetch_weights
        return str(fetch_weights("fvmd"))

    def _load(self):
        if self._model is None:
            import torch
            _ensure_vendor()
            from fvmd.nets.pips2 import Pips
            # PIPs++ uses grid_sample/correlation ops with patchy MPS support: CUDA or CPU.
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = Pips(stride=8).to(self._device)
            state = torch.load(self._resolve_weights(), map_location=self._device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            self._model = model.eval()
        return self._model

    def _run_pips(self, frames):
        """frames [1,S,3,H,W] -> trajectories [1,S,N,2]."""
        import torch
        from fvmd.utils.basic import meshgrid2d
        model = self._model
        B, S, C, H, W = frames.shape
        n = int(np.sqrt(self.n_points).round())
        grid_y, grid_x = meshgrid2d(B, n, n, stack=False, norm=False, device=frames.device)
        grid_y = 8 + grid_y.reshape(B, -1) / float(n - 1) * (H - 16)
        grid_x = 8 + grid_x.reshape(B, -1) / float(n - 1) * (W - 16)
        xy0 = torch.stack([grid_x, grid_y], dim=-1)              # [1, n*n, 2]
        trajs0 = xy0.unsqueeze(1).repeat(1, S, 1, 1)            # [1, S, N, 2]
        preds, _, _, _ = model(trajs0, frames, iters=self.iters, feat_init=None, beautify=True)
        return preds[-1]

    @staticmethod
    def _velocity(trajs):
        import torch
        B, S, N, _ = trajs.shape
        v = trajs[:, 1:] - trajs[:, :-1]
        return torch.cat([torch.zeros(B, 1, N, 2, device=trajs.device), v], dim=1)

    @staticmethod
    def _acceleration(velo):
        import torch
        B, S, N, _ = velo.shape
        a = velo[:, 2:] - velo[:, 1:-1]
        return torch.cat([torch.zeros(B, 2, N, 2, device=velo.device), a], dim=1)

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        from vidaudit.detectors._extract import decode_clip
        model = self._load()                                  # inserts the vendored path
        from fvmd.extract_motion_features import calc_hist     # now importable
        frames = decode_clip(clip.path, n_frames=self.n_frames, size=self.size)  # [S,H,W,3] u8
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            trajs = self._run_pips(t)
            velo = self._velocity(trajs)
            acc = self._acceleration(velo)
        v_hist = calc_hist(velo.detach().cpu().numpy().astype(np.float32)).reshape(-1)
        a_hist = calc_hist(acc.detach().cpu().numpy().astype(np.float32)).reshape(-1)
        return np.concatenate([v_hist, a_hist]).astype(float)
