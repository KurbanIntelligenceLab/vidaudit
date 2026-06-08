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


def _fake_ckpt():
    """A ResNet50 head state dict matching AIGVDet._resnet50_head() (fc = Linear(2048, 1)).
    Built with weights=None so no downloads happen and it loads on CPU in well under a second."""
    import torchvision
    return torchvision.models.resnet50(weights=None, num_classes=1).state_dict()


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


# ---- load_weights() state-dict handling (fake checkpoints, no real weights) -----

def test_load_weights_requires_path():
    """No auto-download weights -> a clear ValueError, not a crash (cf. l3de / stall)."""
    from vidaudit.detectors.aigvdet import AIGVDet
    d = AIGVDet(device="cpu")
    with pytest.raises(ValueError):
        d.load_weights()                 # None
    with pytest.raises(ValueError):
        d.load_weights("")               # empty


def test_load_weights_single_file(tmp_path):
    """A single .pth -> spatial branch only (optical disabled). Plain, module.-prefixed,
    and {"model": sd}-wrapped checkpoints all load."""
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet
    sd = _fake_ckpt()

    plain = tmp_path / "ckpt.pth"
    torch.save(sd, plain)
    d = AIGVDet(device="cpu")
    d.load_weights(str(plain))
    assert d._rgb is not None and d._flow is None       # spatial-only

    prefixed = tmp_path / "prefixed.pth"
    torch.save({"module." + k: v for k, v in sd.items()}, prefixed)
    d = AIGVDet(device="cpu")
    d.load_weights(str(prefixed))                        # "module." stripped
    assert d._rgb is not None and d._flow is None

    wrapped = tmp_path / "wrapped.pth"
    torch.save({"model": sd}, wrapped)
    d = AIGVDet(device="cpu")
    d.load_weights(str(wrapped))                         # unwrapped from "model" key
    assert d._rgb is not None and d._flow is None


def test_load_weights_dir_two_stream(tmp_path):
    """A directory with original.pth + optical.pth -> both branches loaded."""
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet
    torch.save(_fake_ckpt(), tmp_path / "original.pth")
    torch.save(_fake_ckpt(), tmp_path / "optical.pth")
    d = AIGVDet(device="cpu")
    d.load_weights(str(tmp_path))
    assert d._rgb is not None and d._flow is not None


def test_load_weights_dir_spatial_only(tmp_path):
    """A directory with only original.pth -> optical branch stays disabled."""
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet
    torch.save(_fake_ckpt(), tmp_path / "original.pth")
    d = AIGVDet(device="cpu")
    d.load_weights(str(tmp_path))
    assert d._rgb is not None and d._flow is None


def test_load_weights_dir_missing_original(tmp_path):
    """A directory with no original*.pth -> FileNotFoundError (optical alone is not enough)."""
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet
    torch.save(_fake_ckpt(), tmp_path / "optical.pth")
    d = AIGVDet(device="cpu")
    with pytest.raises(FileNotFoundError):
        d.load_weights(str(tmp_path))


def test_load_weights_dir_original_glob_fallback(tmp_path):
    """No exact original.pth -> fall back to the original*.pth glob (e.g. original_best.pth)."""
    import torch
    from vidaudit.detectors.aigvdet import AIGVDet
    torch.save(_fake_ckpt(), tmp_path / "original_best.pth")
    d = AIGVDet(device="cpu")
    d.load_weights(str(tmp_path))
    assert d._rgb is not None and d._flow is None
