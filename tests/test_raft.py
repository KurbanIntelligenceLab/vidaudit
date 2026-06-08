"""RAFT (Teed & Deng, 2020): the eval-path packing + spectral-extractor glue, on synthetic
flow (no torchvision RAFT weights, no GPU, no real video). The RAFT-Large backbone itself
auto-downloads weights and is validated on the cluster; here we exercise everything *around*
the model: _flow_to_mv channel packing + adaptive pooling, the 13-d feature_vector wiring,
and the end-to-end features() path with the model + decode monkeypatched out (cf. test_aigvdet
bypassing RAFT, test_stall exercising the pure scoring math). numpy + torch + scipy + spectrum
only; torchvision is NOT imported.
"""
from __future__ import annotations

import numpy as np
import pytest

from vidaudit.features import FEATURE_NAMES


def _synthetic_flow(T, grid, seed=0):
    """A deterministic [T,2,grid,grid] float32 flow that is BOTH time- and space-varying
    (so every spatial region has nonzero temporal std -> a valid PSD) and strictly positive
    in dx (so |dx| == dx and every frame has nonzero motion -> kept by _identify_iframes)."""
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    flow = np.zeros((T, 2, grid, grid), np.float32)
    for t in range(T):
        flow[t, 0] = 2.0 + 0.3 * np.sin(0.5 * t) + 0.05 * xx     # dx > 0 everywhere
        flow[t, 1] = 1.0 + 0.2 * np.cos(0.3 * t) + 0.05 * yy     # dy
    return flow


# ---- registration / contract (the spec is the only thing previously covered) --------

def test_registered_features_only():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "raft" in all_detectors()
    d = get("raft")
    # RAFT is a frozen, features-only reference: no native head, no training, GPU-preferred.
    assert d.has_features and not d.has_native_head
    assert d.spec.needs_gpu and not d.spec.trainable and not d.spec.published_weights
    assert d.spec.family == "motion"


def test_score_not_implemented():
    """RAFT exposes only features(); the native-head path must raise, not silently return."""
    from vidaudit.detectors.raft import RAFT
    from vidaudit.detectors.base import Clip
    d = RAFT(device="cpu")
    with pytest.raises(NotImplementedError):
        d.score(Clip(video_id="v", path="x", source="g", is_real=0))


# ---- feature_names: the canonical MV-13 schema -------------------------------------

def test_feature_names_are_canonical_mv13():
    from vidaudit.detectors.raft import RAFT
    expected = [
        "mean_mv_mag", "mv_sparsity", "mv_variance", "mv_temporal_diff",
        "spectral_slope_median", "spectral_slope_iqr",
        "spectral_flatness_median", "spectral_flatness_iqr",
        "acf_decay_rate_median", "acf_decay_rate_iqr", "acf_first_zero_median",
        "accel_kurtosis_median", "accel_skewness_median",
    ]
    assert RAFT.feature_names == expected
    assert RAFT.feature_names == FEATURE_NAMES
    assert len(RAFT.feature_names) == 13


# ---- _flow_to_mv: channel packing -------------------------------------------------

def test_flow_to_mv_channel_packing():
    """[T,2,grid,grid] -> [T,grid,grid,5]; ch0 (SOURCE) == 1.0, ch1 == dx, ch2 == dy,
    ch3 == ch4 == 0. Constant-per-frame flow makes every channel hand-checkable."""
    import torch
    from vidaudit.detectors.raft import RAFT
    grid, T = 8, 5
    d = RAFT(grid=grid)
    flow = torch.zeros(T, 2, grid, grid)
    flow[:, 0] = 2.0    # dx
    flow[:, 1] = -3.0   # dy
    out = d._flow_to_mv(flow)

    assert out.shape == (T, grid, grid, 5)
    assert out.dtype == np.float64
    assert np.all(out[..., 0] == 1.0)    # CH_SOURCE marker keeps frames in _identify_iframes
    assert np.all(out[..., 1] == 2.0)    # CH_DST_X == dx
    assert np.all(out[..., 2] == -3.0)   # CH_DST_Y == dy
    assert np.all(out[..., 3] == 0.0)    # CH_SRC_X
    assert np.all(out[..., 4] == 0.0)    # CH_SRC_Y


def test_flow_to_mv_grid_sized_input_not_pooled():
    """When flow is already grid x grid, values pass through verbatim (no pooling distortion)."""
    import torch
    from vidaudit.detectors.raft import RAFT
    grid = 4
    d = RAFT(grid=grid)
    rng = np.random.RandomState(0)
    flow_np = rng.randn(3, 2, grid, grid).astype("float32")
    out = d._flow_to_mv(torch.from_numpy(flow_np))
    assert out.shape == (3, grid, grid, 5)
    assert np.allclose(out[..., 1], flow_np[:, 0])   # dx untouched
    assert np.allclose(out[..., 2], flow_np[:, 1])   # dy untouched


def test_flow_to_mv_adaptive_pool_constant_field():
    """flow H,W != grid -> adaptive_avg_pool2d to grid. A spatially-constant field is
    invariant under average pooling, so the pooled values equal the input constants exactly."""
    import torch
    from vidaudit.detectors.raft import RAFT
    grid = 4
    d = RAFT(grid=grid)
    flow = torch.zeros(2, 2, 16, 16)   # 16x16 -> must pool down to 4x4
    flow[:, 0] = 5.0
    flow[:, 1] = 7.0
    out = d._flow_to_mv(flow)
    assert out.shape == (2, grid, grid, 5)
    assert np.allclose(out[..., 1], 5.0)
    assert np.allclose(out[..., 2], 7.0)


def test_flow_to_mv_adaptive_pool_block_mean():
    """Adaptive avg-pool of a 4x4 -> 2x2 is an exact 2x2 block mean. Build four constant
    blocks (1,2,3,4) so the pooled dx is exactly [[1,2],[3,4]]."""
    import torch
    from vidaudit.detectors.raft import RAFT
    d = RAFT(grid=2)
    f = torch.zeros(1, 2, 4, 4)
    f[0, 0, :2, :2] = 1.0
    f[0, 0, :2, 2:] = 2.0
    f[0, 0, 2:, :2] = 3.0
    f[0, 0, 2:, 2:] = 4.0
    out = d._flow_to_mv(f)
    assert out.shape == (1, 2, 2, 5)
    assert np.array_equal(out[0, :, :, 1], np.array([[1.0, 2.0], [3.0, 4.0]]))


# ---- feature_vector(_flow_to_mv(...), target_frames=n-1): the full extractor glue ----

def test_feature_vector_finite_13d_aligned():
    """_flow_to_mv -> feature_vector(target_frames=n-1) yields a finite 13-d array aligned
    to FEATURE_NAMES (the RAFT detector's feature schema). n_frames=12 -> 11 flow pairs."""
    import torch
    from vidaudit.detectors.raft import RAFT
    grid, n_frames = 8, 12
    T = n_frames - 1                       # 11 consecutive-pair flow maps
    d = RAFT(grid=grid)
    from vidaudit.features import feature_vector
    mv = d._flow_to_mv(torch.from_numpy(_synthetic_flow(T, grid)))
    v = feature_vector(mv, target_frames=n_frames - 1)
    assert v is not None
    assert v.shape == (13,)
    assert v.dtype == np.float64
    assert np.all(np.isfinite(v))


def test_feature_vector_tier0_exact():
    """The Tier-0 stats (mean / sparsity / variance / temporal-diff of per-frame mean
    magnitude) are closed-form. With dy==0 and dx>0 everywhere, magnitude == dx, so we can
    recompute the four Tier-0 features from the synthetic input with plain numpy and match."""
    import torch
    from vidaudit.detectors.raft import RAFT
    from vidaudit.features import feature_vector
    grid, n_frames = 8, 12
    T = n_frames - 1
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    base = 2.0 + 0.3 * np.sin(0.5 * np.arange(T))        # per-frame offset
    ripple = 0.05 * np.sin(xx * 0.7) + 0.05 * np.cos(yy * 0.5)
    flow = np.zeros((T, 2, grid, grid), np.float32)
    for t in range(T):
        flow[t, 0] = base[t] + ripple                    # dx > 0 everywhere -> mag == dx
        flow[t, 1] = 0.0                                 # dy == 0

    d = RAFT(grid=grid)
    mv = d._flow_to_mv(torch.from_numpy(flow))
    v = feature_vector(mv, target_frames=n_frames - 1)
    assert v is not None and np.all(np.isfinite(v))

    # Independent numpy reference for Tier-0 (mag == dx because dy == 0 and dx > 0):
    mag_ref = (base[:, None, None] + ripple[None, :, :]).astype(np.float64)
    global_mag = mag_ref.mean(axis=(1, 2))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert np.isclose(v[idx["mean_mv_mag"]], global_mag.mean())
    assert np.isclose(v[idx["mv_variance"]], np.var(global_mag))
    assert np.isclose(v[idx["mv_temporal_diff"]], np.mean(np.abs(np.diff(global_mag))))
    assert v[idx["mv_sparsity"]] == 0.0                  # no zero-magnitude cells


def test_feature_vector_rejects_too_few_motion_frames():
    """target_frames demands at least that many non-I-frames after filtering. With only 11
    motion frames present, target_frames=12 must be rejected (None), not fabricated."""
    import torch
    from vidaudit.detectors.raft import RAFT
    from vidaudit.features import feature_vector
    grid = 8
    d = RAFT(grid=grid)
    mv = d._flow_to_mv(torch.from_numpy(_synthetic_flow(11, grid)))
    assert feature_vector(mv, target_frames=11) is not None    # exactly enough
    assert feature_vector(mv, target_frames=12) is None        # one short -> reject


def test_feature_vector_constant_flow_returns_none():
    """A spatially- and temporally-constant (but nonzero) flow keeps every frame as a
    motion frame, yet every per-region series has zero variance -> no valid PSD -> the
    extractor returns None rather than emitting silent NaNs."""
    import torch
    from vidaudit.detectors.raft import RAFT
    from vidaudit.features import feature_vector
    grid = 8
    d = RAFT(grid=grid)
    flow = torch.full((11, 2, grid, grid), 3.0)
    mv = d._flow_to_mv(flow)
    assert feature_vector(mv, target_frames=11) is None


# ---- features(clip): end-to-end eval path, model + decode monkeypatched out ---------

def test_features_end_to_end_monkeypatched(monkeypatch):
    """Drive features() with NO RAFT weights / GPU / real video: stub _load to hand back a
    fake flow model + identity transform, and stub the decode so it never touches PyAV.
    The synthetic flow flows through _flow_to_mv + the 13-d extractor to a finite vector."""
    import torch
    import vidaudit.detectors._extract as _extract
    from vidaudit.detectors.raft import RAFT
    from vidaudit.detectors.base import Clip

    grid, n_frames, size = 8, 12, 16
    d = RAFT(n_frames=n_frames, size=size, grid=grid, device="cpu")
    T_pairs = n_frames - 1
    flow_t = torch.from_numpy(_synthetic_flow(T_pairs, grid))

    class _FakeFlowModel:
        def __call__(self, img1, img2, num_flow_updates=12):
            return [flow_t]                       # model(...)[-1] is the final flow estimate

    monkeypatch.setattr(d, "_load", lambda: (_FakeFlowModel(), (lambda a, b: (a, b))))
    monkeypatch.setattr(
        _extract, "decode_clip",
        lambda path, n_frames=8, size=224, window_sec=None: np.zeros((n_frames, size, size, 3), np.uint8),
    )

    c = Clip(video_id="v", path="/no/such/file.mp4", source="g", is_real=0)
    v = d.features(c)
    assert v.shape == (13,)
    assert np.all(np.isfinite(v))


def test_features_requires_two_frames(monkeypatch):
    """A clip that decodes to a single frame cannot form an optical-flow pair: features()
    raises a clear RuntimeError instead of indexing past the end."""
    import vidaudit.detectors._extract as _extract
    from vidaudit.detectors.raft import RAFT
    from vidaudit.detectors.base import Clip

    d = RAFT(n_frames=12, size=8, grid=4, device="cpu")
    monkeypatch.setattr(d, "_load", lambda: (object(), (lambda a, b: (a, b))))
    monkeypatch.setattr(
        _extract, "decode_clip",
        lambda path, n_frames=8, size=224, window_sec=None: np.zeros((1, size, size, 3), np.uint8),
    )
    with pytest.raises(RuntimeError):
        d.features(Clip(video_id="v", path="x", source="g", is_real=0))
