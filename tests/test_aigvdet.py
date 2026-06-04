"""AIGVDet wrapper (arXiv:2403.16638): architecture + pipeline smoke tests with random
weights (no checkpoints downloaded). Real-weights reproduction is a cluster follow-up."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest


def _tiny_clip(path, size=480, dur=1):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={size}x{size}:rate=8:duration={dur}",
                    "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)


def test_registered_with_both_heads():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "aigvdet" in all_detectors()
    d = get("aigvdet")
    assert d.spec.published_weights and d.has_native_head and d.has_features


def test_embed_and_flowviz_shapes():
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet, _flow_to_rgb
    d = AIGVDet(device="cpu")
    feat, logit = AIGVDet._embed(d._ensure(), torch.zeros(2, 3, 64, 64))   # random ResNet50
    assert feat.shape == (2, 2048) and logit.shape == (2, 1)
    img = _flow_to_rgb(np.random.RandomState(0).randn(16, 16, 2).astype("float32"))
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_score_features_spatial_only(tmp_path):
    """No checkpoint loaded -> spatial branch only: score in [0,1], 2048-d features."""
    from vidaudit.detectors.aigvdet import AIGVDet
    from vidaudit.detectors.base import Clip
    mp4 = tmp_path / "c.mp4"
    _tiny_clip(mp4)
    d = AIGVDet(n_frames=4, device="cpu")
    c = Clip(video_id="v", path=str(mp4), source="g", is_real=0)
    s = d.score(c)
    assert 0.0 <= s <= 1.0
    assert d.features(c).shape == (2048,)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_two_stream_fusion(tmp_path, monkeypatch):
    """Optical branch present -> fused score + 4096-d features (RAFT bypassed for speed)."""
    from vidaudit.detectors.aigvdet import AIGVDet
    from vidaudit.detectors.base import Clip
    mp4 = tmp_path / "c.mp4"
    _tiny_clip(mp4)
    d = AIGVDet(n_frames=4, device="cpu")
    d._ensure()
    d._flow = d._resnet50_head()                          # random optical branch
    monkeypatch.setattr(d, "_flow_frames", lambda frames: frames)   # skip RAFT in the unit test
    c = Clip(video_id="v", path=str(mp4), source="g", is_real=0)
    assert 0.0 <= d.score(c) <= 1.0
    assert d.features(c).shape == (4096,)
