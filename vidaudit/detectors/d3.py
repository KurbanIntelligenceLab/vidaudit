"""D3 (Zheng et al., ICCV 2025, arXiv:2508.00701): a training-free detector built on
the second-order temporal difference of XCLIP per-frame embeddings,

    Delta^2 f(t) = f(t+1) - 2 f(t) + f(t-1).

`features(clip)` returns the mean over time of Delta^2 f (768-d), the matched-harness
representation (ported verbatim from the cluster's run_d3_features.py). The XCLIP
backbone auto-downloads, so no manual setup is needed.

Backbone note: the paper specifies XCLIP-ViT-B/16; the original cluster extraction
used B/32. The default here is the paper's B/16; pass variant="B/32" to reproduce the
original feature CSVs exactly.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_HF = {"B/16": "microsoft/xclip-base-patch16", "B/32": "microsoft/xclip-base-patch32"}
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@register("d3")
class D3(Detector):
    spec = DetectorSpec(
        name="D3", backbone="XCLIP-ViT-B/16", family="appearance",
        published_weights=False, trainable=False, needs_gpu=True,
        paper="Zheng et al., ICCV 2025 (arXiv:2508.00701)",
        notes="training-free; XCLIP backbone auto-downloads. Paper uses B/16; the "
              "original extraction used B/32 (variant='B/32' to reproduce it).",
    )

    def __init__(self, variant: str = "B/16", n_frames: int = 8, size: int = 224, device=None):
        self.variant = variant
        self.n_frames = n_frames
        self.size = size
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import XCLIPVisionModel
            self._device = self._device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
            self._model = XCLIPVisionModel.from_pretrained(_HF[self.variant]).to(self._device).eval()
        return self._model

    def _delta2(self, clip: Clip) -> np.ndarray:
        """Per-clip second-order temporal difference of XCLIP embeddings: [S-2, 768]."""
        import torch
        from vidaudit.detectors._extract import decode_clip
        model = self._load()
        frames = decode_clip(clip.path, n_frames=self.n_frames, size=self.size)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(self._device) / 255.0
        mean = torch.tensor(_CLIP_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=self._device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            f = model(pixel_values=x).pooler_output.float().cpu().numpy()  # [S, 768]
        if f.shape[0] < 3:
            raise RuntimeError(f"need >=3 frames for the second-order difference (got {f.shape[0]})")
        return f[2:] - 2.0 * f[1:-1] + f[:-2]

    def features(self, clip: Clip) -> np.ndarray:
        """Mean over time of the second-order difference (768-d)."""
        return self._delta2(clip).mean(axis=0)
