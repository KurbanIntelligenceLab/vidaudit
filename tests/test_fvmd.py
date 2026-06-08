"""FVMD motion baseline (Liu et al., 2024): the pure motion math (velocity / acceleration
first/second differences) and the vendored HOG-style histogram (calc_hist), plus weight
resolution. The PIPs++ point tracker is heavy + GPU-leaning (grid_sample/correlation ops)
and is NOT exercised here; it is validated on the cluster (like the AIGVDet RAFT branch).

All eval-path math runs on synthetic torch/numpy with hand-computed expected values; no
weights, GPU, network, or real video are touched."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# calc_hist lives in the vendored fvmd package; make it importable the same way the
# wrapper does (FVMD._load inserts this onto sys.path before importing it).
_VENDOR = str((Path(__file__).resolve().parents[1] / "vidaudit" / "_vendor"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


# ---- velocity / acceleration: pure static torch on synthetic [1, S, N, 2] ----------

def test_velocity_prepends_zero_and_is_first_difference():
    """_velocity = first differences with a leading zero frame, so frame 0 is exactly
    zero and frame t (t>=1) equals trajs[t] - trajs[t-1]."""
    import torch
    from vidaudit.detectors.fvmd import FVMD
    B, S, N = 1, 5, 3
    trajs = torch.arange(B * S * N * 2, dtype=torch.float32).reshape(B, S, N, 2)
    v = FVMD._velocity(trajs)
    assert v.shape == (B, S, N, 2)
    assert torch.equal(v[:, 0], torch.zeros(B, N, 2))            # zero frame prepended
    assert torch.equal(v[:, 1:], trajs[:, 1:] - trajs[:, :-1])   # rest = first differences


def test_acceleration_prepends_two_zeros_and_is_second_difference():
    """_acceleration = differences of velocity with TWO leading zero frames, so frames
    0 and 1 are exactly zero and frame t (t>=2) equals velo[t] - velo[t-1]."""
    import torch
    from vidaudit.detectors.fvmd import FVMD
    B, S, N = 1, 6, 4
    velo = torch.arange(B * S * N * 2, dtype=torch.float32).reshape(B, S, N, 2)
    a = FVMD._acceleration(velo)
    assert a.shape == (B, S, N, 2)
    assert torch.equal(a[:, :2], torch.zeros(B, 2, N, 2))        # two zero frames prepended
    assert torch.equal(a[:, 2:], velo[:, 2:] - velo[:, 1:-1])    # rest = velocity differences


def test_acceleration_of_velocity_is_second_difference_of_trajectory():
    """Composing _acceleration(_velocity(traj)) on a constant-acceleration trajectory
    (x_t = t^2 along x) yields a constant second difference of 2 for t>=2, zero before."""
    import torch
    from vidaudit.detectors.fvmd import FVMD
    B, S, N = 1, 5, 1
    t = torch.arange(S, dtype=torch.float32)
    traj = torch.zeros(B, S, N, 2)
    traj[0, :, 0, 0] = t * t                                     # x_t = t^2 -> accel == 2
    a = FVMD._acceleration(FVMD._velocity(traj))
    # frame0 (v zero), frame1 (a zero) are seeded zero; thereafter d2/dt2(t^2) == 2.
    assert torch.equal(a[0, :2, 0, 0], torch.zeros(2))
    assert torch.allclose(a[0, 2:, 0, 0], torch.full((S - 2,), 2.0))


# ---- vendored calc_hist on synthetic velocity / accel [1, 16, 400, 2] ---------------

def test_calc_hist_yields_512_per_field_and_1024_concat():
    """calc_hist on a [1,16,400,2] field reshapes 400 -> 20x20 and cuts 5x5 x 4-frame
    subcubes (MS=MH=MW=4) x 8 angle bins = 1*4*4*4*8 = 512 once flattened. Velocity +
    acceleration concatenated give 1024-d, matching len(FVMD.feature_names)."""
    from fvmd.extract_motion_features import calc_hist
    from vidaudit.detectors.fvmd import FVMD
    rng = np.random.RandomState(0)
    velo = rng.randn(1, 16, 400, 2).astype(np.float32)
    acc = rng.randn(1, 16, 400, 2).astype(np.float32)
    assert calc_hist(velo).shape == (1, 4, 4, 4, 8)              # 4*4*4*8 == 512
    v_hist = calc_hist(velo).reshape(-1)
    a_hist = calc_hist(acc).reshape(-1)
    assert v_hist.shape == (512,) and a_hist.shape == (512,)
    feat = np.concatenate([v_hist, a_hist])
    assert feat.shape == (1024,)
    assert len(FVMD.feature_names) == 1024 == feat.shape[0]


def test_calc_hist_directional_velocity_known_bin_and_weight():
    """A uniform velocity field of (x=1, y=0) lands every voxel in one angle bin with a
    hand-computed magnitude weight. magnitude=1 -> log2(1+1)=1 -> ceil -> /log2(256)=1/8
    = 0.125 per voxel; each 4-frame x 5x5 subcube holds 100 voxels -> 12.5 in that bin,
    and 0 in the other seven. 64 subcubes -> total mass 64*12.5 == 800."""
    from fvmd.extract_motion_features import calc_hist
    v = np.zeros((1, 16, 400, 2), np.float32)
    v[..., 0] = 1.0                                              # x-component only
    h = calc_hist(v)                                            # [1,4,4,4,8]
    per_subcube = h.reshape(-1, 8)
    nz = np.flatnonzero(per_subcube[0])
    assert nz.tolist() == [5]                                   # arctan2(1,0) bins to index 5
    assert np.allclose(per_subcube[:, 5], 12.5)                 # 100 voxels * 0.125
    assert np.allclose(np.delete(per_subcube, 5, axis=1), 0.0)  # all other bins empty
    assert np.isclose(h.sum(), 800.0)


def test_calc_hist_static_field_is_all_zero_no_nan():
    """A fully static motion field (every vector exactly zero, the degenerate case for a
    frozen clip) yields an all-zero, finite 1024-d feature: magnitude 0 -> weight 0, no
    NaN/inf despite arctan2(0,0). This is the hand-known value, not a vacuous check."""
    from fvmd.extract_motion_features import calc_hist
    zero = np.zeros((1, 16, 400, 2), np.float32)
    v_hist = calc_hist(zero).reshape(-1)
    a_hist = calc_hist(zero).reshape(-1)
    feat = np.concatenate([v_hist, a_hist]).astype(float)
    assert feat.shape == (1024,)
    assert np.array_equal(feat, np.zeros(1024))                 # exact zeros
    assert np.isfinite(feat).all()                              # no silent NaN/inf


# ---- weight loading + resolution (no real weights, no download) ---------------------

def test_load_weights_sets_path_and_nulls_model():
    """load_weights('p') records the explicit path and invalidates any built model;
    load_weights(None) leaves the prior path intact but STILL nulls the model so the
    next _load rebuilds."""
    from vidaudit.detectors.fvmd import FVMD
    d = FVMD(device="cpu")
    d._model = "STALE"
    d.load_weights("p")
    assert d._weights == "p" and d._model is None
    d._model = "STALE_AGAIN"
    d.load_weights(None)
    assert d._weights == "p" and d._model is None               # path kept, model nulled


def test_resolve_weights_prefers_explicit_path_over_zoo(monkeypatch):
    """_resolve_weights returns the explicit --weights path verbatim and must NOT consult
    the zoo when one is set; with no path it falls back to fetch_weights('fvmd')."""
    import vidaudit.zoo as zoo
    from vidaudit.detectors.fvmd import FVMD

    def _boom(name, **kw):                                       # zoo must not be touched
        raise AssertionError("fetch_weights should not be called when a path is set")

    monkeypatch.setattr(zoo, "fetch_weights", _boom)
    d = FVMD(weights="/explicit/ckpt.pth", device="cpu")
    assert d._resolve_weights() == "/explicit/ckpt.pth"

    sentinel = "/zoo/sentinel/fvmd.pth"
    monkeypatch.setattr(zoo, "fetch_weights", lambda name, **kw: Path(sentinel))
    d2 = FVMD(device="cpu")                                      # no explicit weights
    assert d2._resolve_weights() == sentinel                    # zoo fallback used


# ---- registration / spec contract ---------------------------------------------------

def test_registered_with_features_only():
    """fvmd is registered, exposes features() (the 1024-d motion vector) but no native
    score head, declares published weights + a frozen (non-trainable) GPU reference."""
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "fvmd" in all_detectors()
    d = get("fvmd")
    assert d.has_features and not d.has_native_head
    assert d.spec.published_weights and not d.spec.trainable and d.spec.needs_gpu
