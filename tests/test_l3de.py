"""L3DE: the reimplemented 3D-CNN (`Net`) forward with random weights + small T, and
registration. The DINOv2-G / RAFT / UniDepth-v2 backbones and the real L3DE.pth are a
cluster follow-up (heavy + non-commercial)."""
from __future__ import annotations

import pytest


def test_net_forward_shapes():
    """The three strided streams collapse to T x 8 x 8 and fuse to (feat=128, p_real)."""
    import torch
    from vidaudit.detectors.l3de import _net_module
    T = 2
    net = _net_module(T).eval()
    d2 = torch.zeros(1, 1536, T, 16, 16)     # DINOv2-G patch-token volume
    fl = torch.zeros(1, 2, T, 288, 512)      # optical flow
    dp = torch.zeros(1, 1, T, 288, 512)      # depth
    with torch.no_grad():
        feat, p_real = net([d2, fl, dp])
    assert feat.shape == (1, 128)
    assert p_real.shape == (1, 1)
    assert 0.0 <= float(p_real) <= 1.0       # sigmoid output (p_real)


def test_registered_noncommercial():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401
    assert "l3de" in all_detectors()
    d = get("l3de")
    assert d.has_native_head and d.has_features
    assert d.spec.needs_gpu and d.spec.published_weights
    assert "CC-BY-NC" in d.spec.license      # non-commercial flag surfaced in the zoo


def test_load_weights_requires_path():
    from vidaudit.detectors.registry import get
    d = get("l3de")
    with pytest.raises(ValueError):
        d.load_weights()                     # no auto-download; must point-to-source
