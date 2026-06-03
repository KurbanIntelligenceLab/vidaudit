"""CLIP appearance baseline (Radford et al., 2021): a repurposed reference that uses
per-frame CLIP image embeddings with no temporal modelling. `features(clip)` mean-pools
CLIP-ViT-B/32 image features over 12 frames -> 512-d (ported from the cluster
run_clip_baseline.py); the audited readout reduces it with PCA-13. The backbone
auto-downloads via transformers, so no manual setup is needed.

Because it is appearance-only, it is the canonical "caught by the real-vs-real floor"
case: its features largely encode which dataset the reals came from.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_MODEL = "openai/clip-vit-base-patch32"


@register("clip")
class CLIP(Detector):
    spec = DetectorSpec(
        name="CLIP", backbone="CLIP-ViT-B/32", family="appearance",
        published_weights=False, trainable=False, needs_gpu=False,
        paper="Radford et al., 2021 (CLIP)",
        notes="appearance-only; mean-pooled CLIP image features (PCA-13 in the readout); "
              "backbone auto-downloads.",
    )

    def __init__(self, n_frames: int = 12, device=None):
        self.n_frames = n_frames
        self._device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self._device = self._device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
            self._model = CLIPModel.from_pretrained(_MODEL).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(_MODEL)
        return self._model, self._processor

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        from vidaudit.detectors._extract import decode_pil
        model, processor = self._load()
        frames = decode_pil(clip.path, n_frames=self.n_frames)
        with torch.no_grad():
            inputs = processor(images=frames, return_tensors="pt").to(self._device)
            # explicit CLIP image-embedding path (== get_image_features, but version-stable)
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            feats = model.visual_projection(pooled)  # [S, 512]
        return feats.mean(dim=0).float().cpu().numpy()
