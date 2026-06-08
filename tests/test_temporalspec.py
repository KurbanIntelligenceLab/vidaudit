"""TemporalSpec (ours, WACV 2027): the codec-MV evaluation path on synthetic inputs.

The shipped suite only checks registration (test_detectors.EXPECTED), so the actual
feature_vector / _extract_mv / features() path was untested. These tests exercise it
with hand-built motion-vector arrays and a fake PyAV container -- no H.264, no weights,
no GPU, no network. Expected values are computed by hand from the synthetic signal
(cf. test_aigvdet's shape asserts and test_stall's whitening/log-lik math).

Synthetic signal used by the all-finite cases: every macroblock cell in frame t carries
the same horizontal displacement dx = round(10 + 5*cos(2*pi*t/32)), dy = 0, so each cell's
magnitude is exactly that integer and the 16 spatial regions are identical (every IQR is
then exactly 0 and each median equals the single region's value). The series is
[15,15,15,14,14,13,12,11,10,9,8,7,6,6,5,5,5,5,5,6,6,7,8,9,10,11,12,13,14,14,15,15], with
mean 320/32 = 10 and population variance 13.5625. T = 32 selects the Welch PSD branch.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vidaudit.features import FEATURE_NAMES, feature_vector
from vidaudit.features import mv as mv_module
from vidaudit.detectors.base import Clip
from vidaudit.detectors.temporalspec import TemporalSpec

# Index of each feature in the canonical 13-d vector (FEATURE_NAMES order).
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# --------------------------------------------------------------------------- #
# synthetic motion-vector builders                                            #
# --------------------------------------------------------------------------- #

def _cos_signal(T: int) -> np.ndarray:
    """Integer dx series round(10 + 5*cos(2*pi*t/T)); smooth, strictly positive, varies."""
    t = np.arange(T)
    return (10 + 5 * np.cos(2 * np.pi * t / T)).round().astype(np.int64)


def _uniform_motion_mv(sig: np.ndarray, H: int = 8, W: int = 8) -> np.ndarray:
    """[T, H, W, 5] MV array: every cell of frame t has dx = sig[t], dy = 0.

    Channels are [SOURCE, DST_X, DST_Y, SRC_X, SRC_Y]; setting DST_X = sig with
    SRC_X = SRC_Y = DST_Y = 0 makes the per-cell magnitude exactly |sig[t]|, and
    every spatial region identical.
    """
    T = len(sig)
    mv = np.zeros((T, H, W, 5), dtype=np.int64)
    mv[:, :, :, 1] = sig[:, None, None]   # DST_X = sig per frame
    return mv


# --------------------------------------------------------------------------- #
# fake PyAV container for _extract_mv (no real H.264 decode)                  #
# --------------------------------------------------------------------------- #

# The structured dtype FFmpeg's AVMotionVector / PyAV's MotionVectors.to_ndarray()
# yields (field names per av/sidedata/motionvectors.pyi).
_MV_DTYPE = np.dtype([
    ("source", "i4"), ("w", "u1"), ("h", "u1"),
    ("src_x", "i2"), ("src_y", "i2"),
    ("dst_x", "i2"), ("dst_y", "i2"),
    ("motion_x", "i2"), ("motion_y", "i2"), ("motion_scale", "u2"),
])


def _mv_rows(rows) -> np.ndarray:
    """Build a MOTION_VECTORS structured array. Each row is
    (source, w, h, src_x, src_y, dst_x, dst_y)."""
    arr = np.zeros(len(rows), dtype=_MV_DTYPE)
    for i, (source, w, h, src_x, src_y, dst_x, dst_y) in enumerate(rows):
        arr[i] = (source, w, h, src_x, src_y, dst_x, dst_y, 0, 0, 0)
    return arr


class _FakeMVs:
    def __init__(self, arr):
        self._arr = arr

    def to_ndarray(self):
        return self._arr


class _FakeSideData:
    def __init__(self, mvs):
        self._mvs = mvs

    def get(self, key):
        return self._mvs if key == "MOTION_VECTORS" else None


class _FakeFrame:
    def __init__(self, width, height, mv_arr):
        self.width = width
        self.height = height
        self.side_data = _FakeSideData(_FakeMVs(mv_arr) if mv_arr is not None else None)


class _FakeCodecCtx:
    def __init__(self):
        self.flags2 = 0


class _FakeStream:
    def __init__(self):
        self.codec_context = _FakeCodecCtx()


class _FakeStreams:
    def __init__(self, stream):
        self.video = [stream]


class _FakeContainer:
    def __init__(self, frames):
        self._frames = frames
        self.streams = _FakeStreams(_FakeStream())
        self.closed = False

    def decode(self, stream):
        for f in self._frames:
            yield f

    def close(self):
        self.closed = True


@pytest.fixture
def fake_av(monkeypatch):
    """Install a fake `av` + `av.codec.context` so TemporalSpec._extract_mv decodes
    a caller-supplied list of frames with no real H.264. Returns an installer that
    takes the frame list and yields the (single) FakeContainer for assertions."""
    holder = {}

    fake_av_mod = types.ModuleType("av")
    fake_ctx_mod = types.ModuleType("av.codec.context")

    class _Flags2:
        export_mvs = 1

    fake_ctx_mod.Flags2 = _Flags2

    def _install(frames):
        container = _FakeContainer(frames)
        holder["container"] = container
        fake_av_mod.open = lambda path: container
        return container

    fake_av_mod.codec = types.SimpleNamespace(context=fake_ctx_mod)
    monkeypatch.setitem(sys.modules, "av", fake_av_mod)
    monkeypatch.setitem(sys.modules, "av.codec.context", fake_ctx_mod)
    return _install


# --------------------------------------------------------------------------- #
# feature_vector(): the 13-d spectral extractor (Welch path)                  #
# --------------------------------------------------------------------------- #

def test_feature_vector_welch_finite_13d_in_order():
    """T = 32 motion-everywhere array -> finite 13-d vector in FEATURE_NAMES order.

    Tier-0 channels are exact for the cos signal (mean 10, var 13.5625, no zeros), and
    because all 16 regions are identical every IQR feature is exactly 0. The Welch-branch
    spectral medians are deterministic; assert them as regression values."""
    mv = _uniform_motion_mv(_cos_signal(32))
    v = feature_vector(mv)                                   # default Welch path (T >= 32)

    assert v is not None
    assert v.shape == (13,) and v.dtype == np.float64
    assert np.all(np.isfinite(v))

    # Tier 0 -- exact, hand-computed from the cos signal.
    assert v[IDX["mean_mv_mag"]] == 10.0
    assert v[IDX["mv_sparsity"]] == 0.0                      # no zero-magnitude cells
    assert v[IDX["mv_variance"]] == pytest.approx(13.5625)
    assert v[IDX["mv_temporal_diff"]] == pytest.approx(20.0 / 31.0)  # mean|diff| = 20/31

    # Identical regions -> every IQR is exactly 0.
    assert v[IDX["spectral_slope_iqr"]] == 0.0
    assert v[IDX["spectral_flatness_iqr"]] == 0.0
    assert v[IDX["acf_decay_rate_iqr"]] == 0.0

    # Deterministic spectral medians (Welch branch, all 16 regions equal).
    assert v[IDX["spectral_slope_median"]] == pytest.approx(3.4579875, abs=1e-6)
    assert v[IDX["spectral_flatness_median"]] == pytest.approx(0.04566899, abs=1e-6)
    assert v[IDX["acf_decay_rate_median"]] == pytest.approx(3.0126582, abs=1e-6)
    assert v[IDX["acf_first_zero_median"]] == 7.0
    assert v[IDX["accel_kurtosis_median"]] == pytest.approx(-0.5, abs=1e-9)
    assert v[IDX["accel_skewness_median"]] == pytest.approx(0.0, abs=1e-12)


def test_feature_vector_order_matches_extract_features_dict():
    """feature_vector must lay out exactly d[k] for k in FEATURE_NAMES (no reordering)."""
    mv = _uniform_motion_mv(_cos_signal(32))
    d = mv_module.extract_features(mv)
    v = feature_vector(mv)
    expected = np.array([d[k] for k in FEATURE_NAMES], dtype=float)
    assert np.array_equal(v, expected)


def test_feature_vector_below_min_frames_returns_none():
    """MIN_FRAMES = 7: with only 6 frames carrying motion, n_valid (6) < 7 -> None.

    The array has 10 frames but only the first 6 have nonzero displacement; the rest are
    all-zero (I-frame-like) and dropped by the motion mask."""
    assert mv_module.MIN_FRAMES == 7
    mv = np.zeros((10, 8, 8, 5), dtype=np.int64)
    for i in range(6):                                       # 6 < MIN_FRAMES
        mv[i, :, :, 1] = 5 + i
    assert feature_vector(mv) is None


def test_feature_vector_min_frames_boundary_is_inclusive():
    """Exactly MIN_FRAMES = 7 motion frames is accepted (n_valid >= threshold)."""
    mv = np.zeros((10, 8, 8, 5), dtype=np.int64)
    for i in range(7):                                       # exactly MIN_FRAMES
        mv[i, :, :, 1] = 5 + (i % 3)                         # varies -> non-degenerate PSD
    v = feature_vector(mv)
    assert v is not None and v.shape == (13,)


def test_target_frames_unmet_rejects_for_leakage():
    """Leakage control: target_frames demands an EXACT window; too few motion frames -> None.

    20 motion frames available, target_frames = 25 -> n_valid (20) < 25 -> reject."""
    mv = _uniform_motion_mv(_cos_signal(20))
    assert feature_vector(mv, target_frames=25) is None


def test_target_frames_truncates_and_returns_13d():
    """target_frames = 8 on a 40-motion-frame array truncates to the first 8 and returns 13-d.

    With 40 >= 8 the window is met; only the first 8 non-I-frames are kept, so n_frames_analyzed
    is exactly 8 (the equalised temporal window) yet the output is still the full 13-d vector."""
    mv = _uniform_motion_mv(_cos_signal(40))
    v = feature_vector(mv, target_frames=8)
    assert v is not None and v.shape == (13,)
    d = mv_module.extract_features(mv, target_frames=8)
    assert d["n_frames_analyzed"] == 8
    assert d["n_frames_total"] == 40


# --------------------------------------------------------------------------- #
# _extract_mv(): 16x16 macroblock binning via the fake PyAV container         #
# --------------------------------------------------------------------------- #

def test_extract_mv_macroblock_binning(fake_av):
    """_extract_mv bins MVs into a 16x16 macroblock grid keyed by source position.

    A 32x32 frame -> (32-1)//16 + 1 = 2 macroblocks per axis. One MV at src (0,0) lands in
    cell (0,0); one at src (16,16) lands in cell (1,1). Channels are stored in order
    [SOURCE, DST_X, DST_Y, SRC_X, SRC_Y]."""
    rows = _mv_rows([
        (1, 16, 16, 0, 0, 3, 4),        # src (0,0) -> mb (y=0, x=0)
        (2, 16, 16, 16, 16, 20, 18),    # src (16,16) -> mb (y=1, x=1)
    ])
    frames = [_FakeFrame(32, 32, rows), _FakeFrame(32, 32, rows)]
    container = fake_av(frames)

    grid = TemporalSpec()._extract_mv("dummy.mp4")
    assert grid.shape == (2, 2, 2, 5)                        # [T, H_mb, W_mb, 5]
    assert np.array_equal(grid[0, 0, 0], [1, 3, 4, 0, 0])    # SOURCE,DST_X,DST_Y,SRC_X,SRC_Y
    assert np.array_equal(grid[0, 1, 1], [2, 20, 18, 16, 16])
    assert np.array_equal(grid[0, 0, 1], [0, 0, 0, 0, 0])    # untouched cells stay zero
    assert np.array_equal(grid[0, 1, 0], [0, 0, 0, 0, 0])
    assert container.closed                                  # container.close() in finally


def test_extract_mv_first_wins_and_size_filter(fake_av):
    """First MV wins per cell, and only 16x16 macroblocks are kept.

    Two MVs target cell (0,0): the first (source 7) is binned, the second (source 9) is
    dropped because the cell is already set. A third MV is 8x8 (w != mb), so it is filtered
    out entirely and its cell (1,1) stays zero."""
    rows = _mv_rows([
        (7, 16, 16, 0, 0, 1, 1),        # mb (0,0): first wins
        (9, 16, 16, 0, 0, 5, 5),        # mb (0,0): ignored (cell already set)
        (3, 8, 8, 16, 16, 0, 0),        # w = 8 != 16 -> dropped
    ])
    fake_av([_FakeFrame(32, 32, rows)])
    grid = TemporalSpec()._extract_mv("dummy.mp4")
    assert np.array_equal(grid[0, 0, 0], [7, 1, 1, 0, 0])    # first MV retained
    assert np.array_equal(grid[0, 1, 1], [0, 0, 0, 0, 0])    # 8x8 MV filtered out


def test_extract_mv_no_motion_vectors_yields_zero_grid(fake_av):
    """A frame with no MOTION_VECTORS side data -> an all-zero macroblock grid (no crash)."""
    fake_av([_FakeFrame(32, 32, None)])
    grid = TemporalSpec()._extract_mv("dummy.mp4")
    assert grid.shape == (1, 2, 2, 5)
    assert not grid.any()


# --------------------------------------------------------------------------- #
# features(): end-to-end (decode monkeypatched)                               #
# --------------------------------------------------------------------------- #

def test_features_end_to_end_shape(monkeypatch):
    """features() = _extract_mv -> feature_vector; with decode stubbed it returns the 13-d vector.

    Mirrors the matched-harness contract checked by extract_table (feature_names length 13)."""
    d = TemporalSpec()
    monkeypatch.setattr(d, "_extract_mv", lambda path: _uniform_motion_mv(_cos_signal(32)))
    clip = Clip(video_id="v", path="x.mp4", source="gen", is_real=0)
    v = d.features(clip)
    assert v.shape == (13,)
    assert len(d.feature_names) == 13
    assert np.all(np.isfinite(v))


def test_features_raises_when_too_few_motion_frames(monkeypatch):
    """When feature_vector returns None (< MIN_FRAMES motion), features() raises RuntimeError
    rather than emitting a None row (the wrapper guards the None contract explicitly)."""
    d = TemporalSpec()
    too_short = np.zeros((10, 8, 8, 5), dtype=np.int64)
    for i in range(6):
        too_short[i, :, :, 1] = 5 + i
    monkeypatch.setattr(d, "_extract_mv", lambda path: too_short)
    clip = Clip(video_id="v", path="x.mp4", source="gen", is_real=0)
    with pytest.raises(RuntimeError):
        d.features(clip)


# --------------------------------------------------------------------------- #
# registration / spec sanity (matches the EXPECTED set in test_detectors)     #
# --------------------------------------------------------------------------- #

def test_registered_codec_control():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "temporalspec" in all_detectors()
    d = get("temporalspec")
    assert d.has_features and not d.has_native_head          # features-only detector
    assert not d.spec.needs_gpu and not d.spec.trainable
    assert d.spec.family == "codec" and d.feature_names == FEATURE_NAMES


# --------------------------------------------------------------------------- #
# KNOWN BUG: silent NaN on a valid time-varying input (see bugs_flagged)      #
# --------------------------------------------------------------------------- #

def test_feature_vector_all_finite_on_periodic_input():
    """A periodic sawtooth dx = 5 + (t mod 5) over 32 frames is a perfectly valid, motion-rich
    clip, but its per-region ACF is itself periodic (never a clean exponential decay), so the
    decay-rate fit fails for every region. The fix returns a well-defined finite decay rate of
    0.0 (the series decorrelates immediately, i.e. no temporal persistence) instead of leaking a
    silent NaN, so the full 13-d vector is finite and the two decay entries are exactly 0.0.

    Every spatial region is identical (uniform-motion input), so this is the fully-degenerate
    branch: the median collapses to the single 0.0 fallback and the IQR is 0.0."""
    t = np.arange(32)
    sig = (5 + (t % 5)).astype(np.int64)
    v = feature_vector(_uniform_motion_mv(sig))
    assert v is not None
    assert np.all(np.isfinite(v))                       # no silent NaN leaks into the vector
    assert v[IDX["acf_decay_rate_median"]] == 0.0       # immediate decorrelation -> tau0 = 0
    assert v[IDX["acf_decay_rate_iqr"]] == 0.0
