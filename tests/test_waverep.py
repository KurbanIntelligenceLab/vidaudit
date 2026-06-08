"""WaveRep (Corvi et al., NeurIPS 2025, arXiv:2506.16802): the eval-path math
(center-crop/zero-pad geometry, ImageNet-normalized preprocessing, and the
[mean, median, max]-of-logits feature reduction) with hand-computed expected
values. The DINOv2 ViT-B/14 backbone is heavy + gated and is NOT loaded here:
`_load` is stubbed and `decode_pil` is monkeypatched, so these run on CPU with
no weights, GPU, network, or real video (cf. test_stall.py / test_aigvdet.py)."""
from __future__ import annotations

import numpy as np
import pytest

from vidaudit.detectors.base import Clip
from vidaudit.detectors.waverep import (WaveRep, _IMAGENET_MEAN, _IMAGENET_STD)


# ---- _cc(): one-axis center crop-or-zero-pad offsets (pure int math) ----------

def test_cc_oversized_center_crops():
    """dim > crop -> crop the centered crop x crop window: src starts at (dim-crop)//2,
    dst starts at 0, both lengths == crop."""
    d = WaveRep(crop=16, device="cpu")
    assert d._cc(20) == (2, 16, 0, 16)          # (20-16)//2 == 2
    assert d._cc(33) == (8, 16, 0, 16)          # (33-16)//2 == 8 (odd extra pixel dropped)


def test_cc_exact_is_identity():
    """dim == crop -> no crop, no pad: full copy from 0 to 0."""
    d = WaveRep(crop=16, device="cpu")
    assert d._cc(16) == (0, 16, 0, 16)


def test_cc_undersized_zero_pads():
    """dim < crop -> copy the whole axis into a centered slot: src 0..dim, dst (crop-dim)//2."""
    d = WaveRep(crop=16, device="cpu")
    assert d._cc(10) == (0, 10, 3, 10)          # (16-10)//2 == 3
    assert d._cc(15) == (0, 15, 0, 15)          # (16-15)//2 == 0 (odd extra pad on the far side)


# ---- _preprocess(): crop/pad to [T,3,crop,crop] + ImageNet normalize ----------

def test_preprocess_shape_and_oversized_crop_values():
    """Oversized constant frame: the centered crop x crop window is all 255, so every
    pixel normalizes to (1 - mean)/std per channel; output is [T,3,crop,crop]."""
    import torch
    d = WaveRep(crop=16, device="cpu")
    over = np.full((20, 20, 3), 255, dtype=np.uint8)
    x = d._preprocess([over])
    assert x.shape == (1, 3, 16, 16)
    mean = np.asarray(_IMAGENET_MEAN)
    std = np.asarray(_IMAGENET_STD)
    exp = (1.0 - mean) / std                    # one expected value per channel
    for ch in range(3):
        # the whole 16x16 plane is the constant cropped value
        assert torch.allclose(x[0, ch], torch.full((16, 16), float(exp[ch])), atol=1e-5)


def test_preprocess_undersized_zero_pads_with_imagenet_offset():
    """Undersized constant frame: the 10x10 content lands at [3:13,3:13] as (1-mean)/std,
    while the zero-padded border normalizes to (0-mean)/std (NOT 0) per channel."""
    import torch
    d = WaveRep(crop=16, device="cpu")
    under = np.full((10, 10, 3), 255, dtype=np.uint8)
    x = d._preprocess([under])
    assert x.shape == (1, 3, 16, 16)
    mean = np.asarray(_IMAGENET_MEAN)
    std = np.asarray(_IMAGENET_STD)
    inside = (1.0 - mean) / std
    border = (0.0 - mean) / std
    for ch in range(3):
        # interior content block
        assert np.isclose(x[0, ch, 3, 3].item(), inside[ch], atol=1e-5)
        assert np.isclose(x[0, ch, 12, 12].item(), inside[ch], atol=1e-5)
        # padded corners normalize to -mean/std, not zero
        assert np.isclose(x[0, ch, 0, 0].item(), border[ch], atol=1e-5)
        assert np.isclose(x[0, ch, 15, 15].item(), border[ch], atol=1e-5)
    # the padded border is strictly different from the content -> padding is detectable
    assert not np.isclose(border[0], inside[0])


def test_preprocess_batches_mixed_frames():
    """Two frames of different native sizes both land in a single [2,3,16,16] tensor."""
    d = WaveRep(crop=16, device="cpu")
    frames = [np.full((20, 20, 3), 255, np.uint8), np.full((10, 10, 3), 255, np.uint8)]
    x = d._preprocess(frames)
    assert x.shape == (2, 3, 16, 16)


# ---- features(): [mean, median, max] of per-frame logits -----------------------

class _StubModel:
    """Stands in for the DINOv2 head: ignores pixel content, returns fixed [T,1] logits
    after asserting the preprocessed tensor has the contract shape."""
    def __init__(self, logits, expect_t, crop):
        import torch
        self._logits = torch.as_tensor(logits, dtype=torch.float32)
        self._expect = (expect_t, 3, crop, crop)

    def __call__(self, x):
        assert tuple(x.shape) == self._expect, (tuple(x.shape), self._expect)
        return self._logits.view(-1, 1)


def _fake_frames(n, size=16, val=100):
    from PIL import Image
    return [Image.fromarray(np.full((size, size, 3), val, np.uint8)) for _ in range(n)]


def test_features_mean_median_max(monkeypatch):
    """features() == [mean, median, max] of the per-frame logits. With logits
    [1,3,2,10]: mean=4.0, median=2.5, max=10.0 (median of an even count averages the
    two middle order statistics 2 and 3)."""
    import vidaudit.detectors._extract as _extract
    d = WaveRep(crop=16, device="cpu", n_frames=4)
    d._model = _StubModel([1.0, 3.0, 2.0, 10.0], expect_t=4, crop=16)  # so _load() skips timm
    monkeypatch.setattr(_extract, "decode_pil",
                         lambda path, n_frames=12, window_sec=None: _fake_frames(4))
    c = Clip(video_id="v", path="x.mp4", source="g", is_real=0)
    feat = d.features(c)
    assert feat.shape == (3,)
    assert feat.dtype == np.dtype("float64")
    assert np.allclose(feat, [4.0, 2.5, 10.0])


def test_features_single_frame_collapses(monkeypatch):
    """A single frame -> mean == median == max == that one logit."""
    import vidaudit.detectors._extract as _extract
    d = WaveRep(crop=16, device="cpu", n_frames=1)
    d._model = _StubModel([7.5], expect_t=1, crop=16)
    monkeypatch.setattr(_extract, "decode_pil",
                         lambda path, n_frames=12, window_sec=None: _fake_frames(1))
    c = Clip(video_id="v", path="x.mp4", source="g", is_real=0)
    feat = d.features(c)
    assert np.allclose(feat, [7.5, 7.5, 7.5])


def test_features_negative_logits_polarity(monkeypatch):
    """Max must track the largest (least negative) logit, mean/median the rest:
    [-4,-1,-3] -> mean=-8/3, median=-3, max=-1."""
    import vidaudit.detectors._extract as _extract
    d = WaveRep(crop=16, device="cpu", n_frames=3)
    d._model = _StubModel([-4.0, -1.0, -3.0], expect_t=3, crop=16)
    monkeypatch.setattr(_extract, "decode_pil",
                         lambda path, n_frames=12, window_sec=None: _fake_frames(3))
    c = Clip(video_id="v", path="x.mp4", source="g", is_real=0)
    feat = d.features(c)
    assert np.allclose(feat, [-8.0 / 3.0, -3.0, -1.0])


def test_features_empty_decode_is_degenerate(monkeypatch):
    """Degenerate input (decode yields no frames). The clip cannot be scored (zero-length
    logits), so features() must refuse with a clear RuntimeError rather than emit a silent
    NaN for mean/median or raise a bare zero-size-reduction ValueError from max()."""
    import vidaudit.detectors._extract as _extract
    d = WaveRep(crop=16, device="cpu", n_frames=0)
    d._model = _StubModel([], expect_t=0, crop=16)
    monkeypatch.setattr(_extract, "decode_pil",
                        lambda path, n_frames=12, window_sec=None: [])
    c = Clip(video_id="v", path="x.mp4", source="g", is_real=0)
    with pytest.raises(RuntimeError, match="no frames"):
        d.features(c)


# ---- load_weights() / _resolve_weights() ---------------------------------------

def test_load_weights_sets_path_and_nulls_model():
    """load_weights(path) records the explicit path and drops any built model so it is
    rebuilt lazily; _resolve_weights() then returns that path verbatim (no zoo fetch)."""
    d = WaveRep(crop=16, device="cpu")
    d._model = object()                          # pretend a model is already built
    d.load_weights("/tmp/g4.pth")
    assert d._weights == "/tmp/g4.pth"
    assert d._model is None
    assert d._resolve_weights() == "/tmp/g4.pth"


def test_load_weights_none_keeps_path_but_nulls_model():
    """load_weights() with no arg keeps the existing explicit path and still nulls the
    model (so a re-load picks up a changed checkpoint on disk)."""
    d = WaveRep(weights="/tmp/prev.pth", crop=16, device="cpu")
    d._model = object()
    d.load_weights()
    assert d._weights == "/tmp/prev.pth"         # unchanged
    assert d._model is None
    assert d._resolve_weights() == "/tmp/prev.pth"


# ---- registration / spec (mirrors the exemplars' final smoke test) -------------

def test_registered():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "waverep" in all_detectors()
    d = get("waverep")
    assert d.has_features and not d.has_native_head      # WaveRep exposes features() only
    assert d.spec.published_weights and d.spec.needs_gpu and not d.spec.trainable
    assert d.feature_names == ["waverep_score_mean", "waverep_score_median", "waverep_score_max"]
