"""ReStraV (Internò et al., NeurIPS 2025, arXiv:2507.00583): the perceptual-straightening
trajectory geometry that turns a frozen DINOv2-S token trajectory Z[B, T, D] into the 21-d
feature vector. The DINOv2 backbone auto-downloads via torch.hub and is heavy + GPU-leaning,
so it is exercised on the cluster (like the STALL / MLLM wrappers); here we drive the pure
geometry math (_moment4 / _features_from_Z) and the trainable-head guard on synthetic
trajectories with hand-computed expected values. _load() and the decode path are never hit."""
from __future__ import annotations

import numpy as np
import pytest


def _straight_line_Z(T=8, D=3, step=2.0):
    """A perfectly straight constant-speed trajectory: each frame advances `step` along
    one axis. Turning angles are all 0 deg; every step distance equals `step`."""
    import torch
    Z = torch.zeros(1, T, D)
    for t in range(T):
        Z[0, t, 0] = step * t
    return Z


def _L_path_Z(T=8, D=3):
    """An L-shaped unit-speed path: 4 steps east then (T-5) steps north. Exactly one 90 deg
    turn at the corner (step index 3 in the T-2 turning-angle sequence); all step distances
    are 1.0."""
    import torch
    Z = torch.zeros(1, T, D)
    pos = torch.zeros(D)
    for t in range(1, T):
        if t <= 4:
            pos[0] += 1.0
        else:
            pos[1] += 1.0
        Z[0, t] = pos.clone()
    return Z


# ---- _moment4: [mean, amin, amax, var(unbiased=False)] stacked --------------------

def test_moment4_against_numpy():
    """_moment4(x) stacks [mean, min, max, population-variance] along dim=1."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [10.0, 10.0, 10.0, 10.0]])
    out = ReStraV._moment4(x)
    assert out.shape == (2, 4)
    # row 0: mean 2.5, min 1, max 4, population var 1.25 (np.var default ddof=0)
    assert np.allclose(out[0].numpy(), [2.5, 1.0, 4.0, 1.25])
    # row 1: constant -> mean=min=max=10, var=0
    assert np.allclose(out[1].numpy(), [10.0, 10.0, 10.0, 0.0])
    # cross-check the variance term is the *biased* (unbiased=False) estimator, not ddof=1
    xn = x.numpy()
    assert np.allclose(out[:, 0].numpy(), xn.mean(axis=1))
    assert np.allclose(out[:, 3].numpy(), xn.var(axis=1))          # ddof=0
    assert not np.allclose(out[:, 3].numpy(), xn.var(axis=1, ddof=1))


# ---- _features_from_Z: documented 21-d layout on known geometries -----------------

def test_features_layout_and_shape_straight_line():
    """A straight constant-speed line (step=2, T=8): step distances all 2.0, every turning
    angle 0 deg. Layout is [d[:7], theta[:6], moment4(d), moment4(theta)] -> 7+6+4+4 = 21."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    Z = _straight_line_Z(T=8, D=3, step=2.0)
    v = ReStraV()._features_from_Z(Z)
    assert v.shape == (1, 21)
    assert torch.isfinite(v).all()
    row = v[0].numpy()
    expected = np.array(
        [2.0] * 7 +                 # d[:7]: seven unit steps of length 2
        [0.0] * 6 +                 # theta[:6]: no turning on a straight line
        [2.0, 2.0, 2.0, 0.0] +      # moment4(d): mean/min/max=2, var=0
        [0.0, 0.0, 0.0, 0.0],       # moment4(theta): all-zero angles
        dtype=float)
    assert np.allclose(row, expected, atol=1e-5)


def test_features_L_path_turning_angle_is_90deg():
    """An L-shaped unit path has exactly one 90 deg turn at the corner. The turning-angle
    block [theta[:6]] must read [0,0,0,90,0,0], and the angle moments follow from it."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    Z = _L_path_Z(T=8, D=3)
    v = ReStraV()._features_from_Z(Z)
    assert v.shape == (1, 21)
    assert torch.isfinite(v).all()
    row = v[0].numpy()
    # step distances: seven unit steps
    assert np.allclose(row[:7], [1.0] * 7, atol=1e-5)
    # turning angles: a single 90 deg turn at the 4th step
    assert np.allclose(row[7:13], [0.0, 0.0, 0.0, 90.0, 0.0, 0.0], atol=1e-4)
    # moment4(d): all steps length 1 -> mean/min/max=1, var=0
    assert np.allclose(row[13:17], [1.0, 1.0, 1.0, 0.0], atol=1e-5)
    # moment4(theta) over [0,0,0,90,0,0]: mean=15, min=0, max=90, pop-var=1125
    assert np.allclose(row[17:21], [15.0, 0.0, 90.0, 1125.0], atol=1e-3)


def test_features_batched_rows_are_independent():
    """[B, T, D] -> [B, 21]; stacking the straight line and the L path in one batch must
    reproduce exactly the per-clip rows (no cross-talk between batch elements)."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    det = ReStraV()
    Z = torch.cat([_straight_line_Z(), _L_path_Z()], dim=0)
    v = det._features_from_Z(Z)
    assert v.shape == (2, 21)
    assert torch.isfinite(v).all()
    assert np.allclose(v[0].numpy(), det._features_from_Z(_straight_line_Z())[0].numpy(),
                       atol=1e-5)
    assert np.allclose(v[1].numpy(), det._features_from_Z(_L_path_Z())[0].numpy(), atol=1e-5)
    # the two geometries differ: same step layout but different turning angles
    assert np.allclose(v[0, 7:13].numpy(), 0.0)                 # straight: no turning
    assert np.isclose(v[1, 10].numpy(), 90.0)                  # L: the corner


# ---- degenerate input: zero-norm steps (static / duplicated frames) ----------------

def test_static_trajectory_has_no_spurious_turning_angle():
    """A duplicated-frame trajectory does not move, so the turning angle is undefined.
    Regression guard for the eps-guarded-cosine bug: F.cosine_similarity returned 0.0 for
    zero-norm steps, so acos(0)=90deg silently encoded a confident right angle at every
    transition of a path that never moves. The degenerate guard now defines theta=0 (no
    turning) wherever a step has no length, so the turning-angle block is all zeros and the
    whole vector stays finite."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    Z = torch.full((1, 8, 3), 0.3)            # all frames identical -> every step is zero
    v = ReStraV()._features_from_Z(Z)
    assert torch.isfinite(v).all()
    # step distances are genuinely zero (this part was always correct)
    assert np.allclose(v[0, :7].numpy(), 0.0, atol=1e-6)
    # a non-moving path has no turns: the turning-angle block is exactly 0, not 90deg
    assert np.allclose(v[0, 7:13].numpy(), 0.0, atol=1e-6)
    # moment4(theta) over an all-zero angle sequence is [mean, min, max, var] = all zeros
    assert np.allclose(v[0, 17:21].numpy(), 0.0, atol=1e-6)


# ---- trainable-head guard: build_model requires cfg.extra["in_dim"] ----------------

def test_build_model_requires_in_dim():
    """The trainer sets cfg.extra['in_dim'] from the feature table before build_model();
    an unset in_dim must raise ValueError, not silently build a zero-width head."""
    from vidaudit.detectors.restrav import ReStraV
    from vidaudit.train.config import TrainConfig
    det = ReStraV()
    cfg = det.default_train_config()              # head='mlp', hidden=(64,), no in_dim
    assert "in_dim" not in cfg.extra
    with pytest.raises(ValueError):
        det.build_model(cfg)


def test_build_model_with_in_dim_builds_21d_head():
    """With cfg.extra['in_dim']=21 the head accepts the 21-d ReStraV feature vector and
    emits one logit per clip (generated-positive convention)."""
    import torch
    from vidaudit.detectors.restrav import ReStraV
    det = ReStraV()
    cfg = det.default_train_config()
    cfg.extra["in_dim"] = 21
    model = det.build_model(cfg)
    out = model(torch.zeros(4, 21))               # batch of 4 21-d feature vectors
    assert out.shape == (4,)                      # one logit per clip


# ---- spec / capability contract (eval-path metadata the harness reads) -------------

def test_default_train_config_is_restrav_recipe():
    """ReStraV ships its own light recipe: a single 64-wide MLP head trained for more epochs."""
    from vidaudit.detectors.restrav import ReStraV
    cfg = ReStraV().default_train_config()
    assert cfg.head == "mlp" and cfg.hidden == (64,)
    assert cfg.loss == "bce" and cfg.epochs == 80


def test_registered_features_only_trainable():
    """Registered as a features() detector (no native score head) and trainable; backbone
    has no published external weights (DINOv2 auto-downloads via torch.hub)."""
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "restrav" in all_detectors()
    d = get("restrav")
    assert d.has_features and not d.has_native_head
    assert d.is_trainable and d.spec.trainable
    assert not d.spec.published_weights and not d.has_pretrained
