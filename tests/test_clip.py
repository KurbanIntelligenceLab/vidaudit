"""CLIP appearance baseline (Radford et al., 2021): the features() eval-path wiring
(decode_pil -> processor -> vision_model.pooler_output -> visual_projection -> mean over
frames -> 512-d), exercised with stub model/processor and synthetic frames. The real
CLIP-ViT-B/32 backbone is heavy + auto-downloads, so it is NOT loaded here (it is
validated on the cluster, like the MLLM / NSG-VD wrappers); we inject deterministic stubs
so the per-frame mean-pool arithmetic is checked against hand-computed values.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vidaudit.detectors.base import Clip


# ---- deterministic stubs (no transformers download, no torch model) -------------

class _StubVisionOut:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _StubModel:
    """vision_model(pixel_values=X) -> .pooler_output == X (passthrough of the per-frame
    rows), and visual_projection == identity. So features() reduces to mean over frames of
    the rows we feed in via the stub processor: a fully hand-computable pipeline."""

    def vision_model(self, pixel_values):
        return _StubVisionOut(pixel_values)

    def visual_projection(self, pooled):
        return pooled                      # identity projection -> output dim == input dim


class _StubBatch(dict):
    """Mimics a transformers BatchEncoding: a dict that also supports .to(device)
    (features() calls processor(...).to(self._device) before indexing pixel_values)."""

    def to(self, device):
        self.device = device
        return self


class _StubProcessor:
    """processor(images=frames, ...) -> {"pixel_values": [S, 512]} where row i is all-`i`.

    Each of the S frames maps to a constant row equal to its frame index, so the mean over
    S frames is, per dim, (0 + 1 + ... + (S-1)) / S == (S - 1) / 2 -- a known scalar."""

    def __init__(self):
        self.last_n = None

    def __call__(self, images, return_tensors=None):
        self.last_n = len(images)
        rows = [torch.full((512,), float(i)) for i in range(len(images))]
        return _StubBatch(pixel_values=torch.stack(rows, dim=0))     # [S, 512], .to()-able


def _make_detector(n_frames, fake_frames, monkeypatch):
    """Build a CLIP detector with stubs injected, decode_pil patched to return
    `fake_frames`, and _load() neutered so no real backbone is touched. Returns
    (detector, processor) so tests can inspect what the processor saw."""
    from vidaudit.detectors import clip as clip_mod

    proc = _StubProcessor()
    d = clip_mod.CLIP(n_frames=n_frames, device="cpu")
    d._model = _StubModel()
    d._processor = proc
    d._device = "cpu"
    # _load must be a no-op that returns the injected stubs (never download CLIP-ViT-B/32).
    monkeypatch.setattr(d, "_load", lambda: (d._model, d._processor))

    captured = {}

    def fake_decode_pil(path, n_frames):       # signature mirrors the real decode_pil call site
        captured["path"] = path
        captured["n_frames"] = n_frames
        return list(fake_frames)

    # patch the symbol where features() imports it from (vidaudit.detectors._extract)
    monkeypatch.setattr("vidaudit.detectors._extract.decode_pil", fake_decode_pil)
    return d, proc, captured


# ---- spec / capabilities --------------------------------------------------------

def test_registered_features_only():
    """clip exposes features() for the matched-harness readout but no native head, and is a
    frozen (non-trainable, no published weights, CPU-ok) appearance reference."""
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "clip" in all_detectors()
    d = get("clip")
    assert d.has_features and not d.has_native_head
    assert not d.spec.published_weights and not d.spec.trainable and not d.spec.needs_gpu
    assert d.spec.family == "appearance" and d.spec.backbone == "CLIP-ViT-B/32"


# ---- features() wiring + hand-computed mean-pool --------------------------------

def test_features_is_512d_float_vector(monkeypatch):
    """features() returns a flat 512-d float32 vector (one mean-pooled CLIP embedding)."""
    d, _, _ = _make_detector(n_frames=3, fake_frames=["f0", "f1", "f2"], monkeypatch=monkeypatch)
    out = d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert isinstance(out, np.ndarray)
    assert out.shape == (512,)
    assert out.dtype == np.float32          # features() ends in .float().cpu().numpy()


def test_features_mean_over_frames(monkeypatch):
    """With S=3 frames whose per-frame projection rows are 0, 1, 2 (every dim), the mean over
    frames is (0+1+2)/3 == 1.0 in every one of the 512 dims. Checks the vision_model ->
    visual_projection -> mean(dim=0) wiring against an exact hand-computed value."""
    d, _, _ = _make_detector(n_frames=3, fake_frames=["a", "b", "c"], monkeypatch=monkeypatch)
    out = d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert np.allclose(out, np.full(512, 1.0))      # (S-1)/2 == 1.0 for S=3


def test_features_mean_changes_with_frame_count(monkeypatch):
    """Same stub, S=5 frames -> rows 0..4 -> mean (S-1)/2 == 2.0 in every dim. Confirms the
    pooling really averages over the decoded frames (not, say, a single frame)."""
    d, _, _ = _make_detector(n_frames=5,
                             fake_frames=["a", "b", "c", "d", "e"], monkeypatch=monkeypatch)
    out = d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert np.allclose(out, np.full(512, 2.0))      # (S-1)/2 == 2.0 for S=5


def test_n_frames_forwarded_to_decode_pil(monkeypatch):
    """The n_frames ctor arg is passed straight through to decode_pil (default is 12, here 7),
    and the clip path is forwarded unchanged."""
    d, proc, captured = _make_detector(
        n_frames=7, fake_frames=[f"f{i}" for i in range(7)], monkeypatch=monkeypatch)
    d.features(Clip(video_id="v", path="/some/clip.mp4", source="g", is_real=0))
    assert captured["n_frames"] == 7
    assert captured["path"] == "/some/clip.mp4"
    assert proc.last_n == 7                          # processor saw all 7 decoded frames


def test_default_n_frames_is_12(monkeypatch):
    """The documented default frame budget is 12 (matches the cluster run_clip_baseline.py)."""
    from vidaudit.detectors import clip as clip_mod
    d = clip_mod.CLIP()
    assert d.n_frames == 12

    d2, _, captured = _make_detector(
        n_frames=12, fake_frames=[f"f{i}" for i in range(12)], monkeypatch=monkeypatch)
    d2.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert captured["n_frames"] == 12
    # rows 0..11 -> mean (S-1)/2 == 5.5
    out = d2.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert np.allclose(out, np.full(512, 5.5))


def test_zero_frames_is_not_nan(monkeypatch):
    """A clip that decodes to zero frames must not silently produce a NaN feature vector.
    There is no sensible mean-pooled embedding for "no frames", so features() now raises a
    clear RuntimeError instead of mean-pooling over an empty dim (which yielded silent NaN)."""
    d, _, _ = _make_detector(n_frames=0, fake_frames=[], monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError, match="zero decoded frames"):
        d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
