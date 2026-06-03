"""WaveRep (Corvi et al., NeurIPS 2025, arXiv:2506.16802): a DINOv2 ViT-B/14 forensic
detector fine-tuned with wavelet-band augmentation. Ported from run_waverep_features.py.

Per frame: center-crop / zero-pad to 504, ImageNet-normalize, forward -> 1-d logit.
`features(clip)` = [mean, median, max] of the logit over 16 frames in a 2s window (the
published-scalar path; matched-harness LOGO-OOD ~0.996). Needs the G4 checkpoint:
`load_weights()` fetches it from the zoo (funded storage) or takes a local path.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
_CROP = 504
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@register("waverep")
class WaveRep(Detector):
    spec = DetectorSpec(
        name="WaveRep", backbone="DINOv2 ViT-B/14 + wavelet aug", family="forensic",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="see zoo/manifest.yaml (waverep)",
        weights_sha256="50d639049d928986ba7d69861a4fe4f3e7afbba1843e3e089cdf6b4748f53d5b",
        paper="Corvi et al., NeurIPS 2025 (arXiv:2506.16802)",
        notes="needs the G4 checkpoint; load_weights() fetches from the zoo or a local path.",
    )
    feature_names = ["waverep_score_mean", "waverep_score_median", "waverep_score_max"]

    def __init__(self, weights=None, n_frames: int = 16, window_sec: float = 2.0,
                 crop: int = _CROP, device=None):
        self._weights = weights
        self.n_frames = n_frames
        self.window_sec = window_sec
        self.crop = crop
        self._device = device
        self._model = None

    def load_weights(self, path=None) -> None:
        if path:
            self._weights = path
        self._model = None  # rebuilt lazily on next use

    def _resolve_weights(self) -> str:
        if self._weights:
            return str(self._weights)
        from vidaudit.zoo import fetch_weights
        return str(fetch_weights("waverep"))

    def _load(self):
        if self._model is None:
            import timm
            import torch
            self._device = self._device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
            model = timm.create_model(_ARCH, num_classes=1, pretrained=False, img_size=self.crop)
            dat = torch.load(self._resolve_weights(), map_location="cpu", weights_only=False)
            if isinstance(dat, dict) and "state_dict" in dat:
                dat = {k[6:]: v for k, v in dat["state_dict"].items() if k.startswith("model.")}
            model.load_state_dict(dat, strict=True)  # raises on any mismatch
            self._model = model.to(self._device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    def _cc(self, dim: int):
        """One-axis center crop-or-zero-pad to self.crop: (src_start, len, dst_start, len)."""
        t = self.crop
        if dim >= t:
            return (dim - t) // 2, t, 0, t
        return 0, dim, (t - dim) // 2, dim

    def _preprocess(self, frames):
        import torch
        out = torch.zeros(len(frames), 3, self.crop, self.crop)
        for i, im in enumerate(frames):
            t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0  # [3,H,W] (copy -> writable)
            sh0, sh, dh0, dh = self._cc(t.shape[1])
            sw0, sw, dw0, dw = self._cc(t.shape[2])
            out[i, :, dh0:dh0 + dh, dw0:dw0 + dw] = t[:, sh0:sh0 + sh, sw0:sw0 + sw]
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return ((out - mean) / std).to(self._device)

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        from vidaudit.detectors._extract import decode_pil
        model = self._load()
        frames = decode_pil(clip.path, n_frames=self.n_frames, window_sec=self.window_sec)
        x = self._preprocess(frames)
        with torch.no_grad():
            logits = model(x).view(-1).float().cpu().numpy()  # [T]
        return np.array([logits.mean(), float(np.median(logits)), logits.max()], dtype=float)
