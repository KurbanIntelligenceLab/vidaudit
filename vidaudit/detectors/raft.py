"""RAFT optical-flow baseline (Teed & Deng, 2020): a repurposed motion reference.

Ported from the cluster run_raft_features.py: decode 12 frames at 256x256, run
RAFT-Large (C_T_SKHT_V2) on the 11 consecutive pairs, average-pool the flow to a 32x32
lattice, pack it as a motion-vector array, and run it through the shared 13-d
TemporalSpec extractor (vidaudit.features). torchvision auto-downloads the RAFT weights.

`feature_names` are the canonical MV-13 columns, so the output matches the codec-MV
schema for an apples-to-apples comparison.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register
from vidaudit.features import FEATURE_NAMES, feature_vector


@register("raft")
class RAFT(Detector):
    spec = DetectorSpec(
        name="RAFT", backbone="RAFT-Large optical flow", family="motion",
        published_weights=False, trainable=False, needs_gpu=True,
        paper="Teed & Deng, 2020 (RAFT)",
        notes="repurposed reference; optical-flow magnitude through the shared 13-d "
              "TemporalSpec spectral extractor. torchvision weights auto-download.",
    )
    feature_names = FEATURE_NAMES

    def __init__(self, n_frames: int = 12, size: int = 256, grid: int = 32,
                 iters: int = 12, device=None):
        self.n_frames = n_frames
        self.size = size
        self.grid = grid
        self.iters = iters
        self._device = device
        self._model = None
        self._tf = None

    def _load(self):
        if self._model is None:
            import torch
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            # RAFT has ops with patchy MPS support; use CUDA when present, else CPU.
            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            weights = Raft_Large_Weights.C_T_SKHT_V2
            self._model = raft_large(weights=weights).to(self._device).eval()
            self._tf = weights.transforms()
        return self._model, self._tf

    def _flow_to_mv(self, flow) -> np.ndarray:
        """[T,2,H,W] pixel flow -> [T,grid,grid,5] MV array (SOURCE, dx, dy, 0, 0)."""
        import torch.nn.functional as F
        T = flow.shape[0]
        if flow.shape[-2:] != (self.grid, self.grid):
            flow = F.adaptive_avg_pool2d(flow, output_size=(self.grid, self.grid))
        dx = flow[:, 0].numpy().astype(np.float64)
        dy = flow[:, 1].numpy().astype(np.float64)
        out = np.zeros((T, self.grid, self.grid, 5), dtype=np.float64)
        out[..., 0] = 1.0   # SOURCE marker (keeps the frame in _identify_iframes)
        out[..., 1] = dx
        out[..., 2] = dy
        return out

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        from vidaudit.detectors._extract import decode_clip
        model, tf = self._load()
        frames = decode_clip(clip.path, n_frames=self.n_frames, size=self.size)  # [S,H,W,3] u8
        t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [S,3,H,W] uint8
        if t.shape[0] < 2:
            raise RuntimeError("need >=2 frames for optical flow")
        img1, img2 = tf(t[:-1].float(), t[1:].float())
        with torch.no_grad():
            flow = model(img1.to(self._device), img2.to(self._device),
                         num_flow_updates=self.iters)[-1].detach().cpu().float()
        mv = self._flow_to_mv(flow)
        v = feature_vector(mv, target_frames=self.n_frames - 1)
        if v is None:
            raise RuntimeError("extract_features returned None (too few motion frames)")
        return v
