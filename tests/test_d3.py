"""D3 (Zheng et al., ICCV 2025, arXiv:2508.00701): the training-free scoring math on the
second-order temporal difference of XCLIP embeddings,

    Delta^2 f(t) = f(t+1) - 2 f(t) + f(t-1),

exercised end-to-end through features()/score() WITHOUT the XCLIP backbone, GPU, network,
or real video. The heavy XCLIP-ViT-B forward is validated on the cluster (like the STALL /
MLLM wrappers); here we monkeypatch D3._load to a fake module that emits a KNOWN
pooler_output [S,768], and monkeypatch decode_clip so no mp4 is touched. Every assertion is
a hand-computed value.
"""
from __future__ import annotations

import numpy as np
import pytest

from vidaudit.detectors.base import Clip
from vidaudit.detectors.d3 import _HF, D3


# ----------------------------------------------------------------------------- helpers

class _FakeOut:
    """Mimics the HF model output: only `.pooler_output` is read by D3._delta2."""

    def __init__(self, pooler):
        import torch
        self.pooler_output = torch.as_tensor(pooler, dtype=torch.float32)


class _FakeModel:
    """Stand-in for XCLIPVisionModel: ignores pixel_values and returns a fixed [S,768]
    pooler_output, so the difference math is driven by KNOWN embeddings, not a real net."""

    def __init__(self, pooler):
        self._pooler = pooler

    def __call__(self, pixel_values=None, **kw):
        return _FakeOut(self._pooler)


def _patch(monkeypatch, d, pooler):
    """Wire a D3 instance to a fake backbone + fake decoder.

    `pooler` is the [S,768] embedding matrix the fake XCLIP "returns". decode_clip is
    replaced with a synthetic uint8 frame stack of matching length so the permute/normalize
    preamble in _delta2 runs on valid input (its result is discarded by the fake model)."""
    import vidaudit.detectors._extract as _extract
    S = int(np.asarray(pooler).shape[0])
    d._device = "cpu"
    monkeypatch.setattr(d, "_load", lambda: _FakeModel(np.asarray(pooler, dtype=np.float32)))
    monkeypatch.setattr(
        _extract, "decode_clip",
        lambda path, n_frames=8, size=224, window_sec=None:
            np.zeros((S, size, size, 3), dtype=np.uint8))


def _clip():
    return Clip(video_id="v", path="/does/not/exist.mp4", source="g", is_real=0)


def _cubic_pooler(S, n_dim=768):
    """f[t] = t**3 broadcast across all 768 dims. For S=5 the interior second differences
    are exactly [6, 12, 18] in every dim (see the hand-computation in the tests)."""
    col = (np.arange(S, dtype=np.float64) ** 3).reshape(S, 1)
    return np.broadcast_to(col, (S, n_dim)).copy()


# ----------------------------------------------------------------------------- registration

def test_registered():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "d3" in all_detectors()
    d = get("d3")
    assert d.has_native_head and d.has_features          # both evidence interfaces
    assert d.spec.needs_gpu and not d.spec.trainable      # training-free, GPU backbone
    assert not d.spec.published_weights                   # XCLIP auto-downloads, no D3 ckpt


# ----------------------------------------------------------------------------- features()

def test_features_is_mean_of_second_difference(monkeypatch):
    """features() == mean over time of Delta^2 f. With f[t]=t**3 (S=5) the second
    differences are [6,12,18] per dim, whose mean is 12.0 in all 768 dims."""
    d = D3(device="cpu")
    _patch(monkeypatch, d, _cubic_pooler(5))
    feat = d.features(_clip())
    assert feat.shape == (768,)
    assert np.allclose(feat, 12.0)


def test_features_matches_explicit_formula_per_dim(monkeypatch):
    """Non-degenerate per-dim check: a hand-built [4,768] pooler with distinct columns
    must reproduce mean(f[2:] - 2 f[1:-1] + f[:-2], axis=0) exactly."""
    rng = np.random.RandomState(0)
    f = rng.randn(4, 768).astype(np.float32)             # S=4 -> d2 [2,768]; matches the
    d = D3(device="cpu")                                 # float32 pooler_output in _delta2
    _patch(monkeypatch, d, f)
    expected = (f[2:] - 2.0 * f[1:-1] + f[:-2]).mean(axis=0)
    assert np.allclose(d.features(_clip()), expected, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------- score()

def test_score_is_negated_std_mean(monkeypatch):
    """score() == -mean_dim(std_time(Delta^2 f, ddof=1)), INCLUDING the sign flip.
    For f[t]=t**3 (S=5): d2=[6,12,18] per dim; std(ddof=1)=6.0; mean over dims=6.0;
    negated -> -6.0 (higher = more generated)."""
    d = D3(device="cpu")
    _patch(monkeypatch, d, _cubic_pooler(5))
    s = d.score(_clip())
    assert isinstance(s, float)
    assert s == pytest.approx(-6.0)


def test_score_sign_is_flipped(monkeypatch):
    """The published convention is higher = more real; D3 negates it. A non-zero temporal
    spread therefore yields a strictly NEGATIVE native score (never the raw +std)."""
    rng = np.random.RandomState(1)
    f = rng.randn(6, 768).astype(np.float32)             # S=6 -> d2 [4,768]; float32 to
    d = D3(device="cpu")                                 # mirror the pooler_output dtype
    _patch(monkeypatch, d, f)
    d2 = f[2:] - 2.0 * f[1:-1] + f[:-2]
    raw = float(d2.std(axis=0, ddof=1).mean())
    assert raw > 0.0
    assert d.score(_clip()) == pytest.approx(-raw, rel=1e-5)   # exactly the negation
    assert d.score(_clip()) < 0.0


# ----------------------------------------------------------------------------- frame-count guard

def test_too_few_frames_raises(monkeypatch):
    """<3 frames cannot form a second-order difference -> documented RuntimeError."""
    d = D3(device="cpu")
    _patch(monkeypatch, d, _cubic_pooler(2))             # only 2 embeddings
    with pytest.raises(RuntimeError, match="need >=3 frames"):
        d.features(_clip())
    with pytest.raises(RuntimeError, match="need >=3 frames"):
        d.score(_clip())


def test_three_frames_features_well_defined(monkeypatch):
    """Exactly 3 frames is the minimum that does NOT raise: d2 has a single row, so the
    MEAN-over-time (features) is just that row and is fully well defined.
    f[t]=t**3 at S=3 -> d2 = f[2]-2 f[1]+f[0] = 8-2+0 = 6 in every dim."""
    d = D3(device="cpu")
    _patch(monkeypatch, d, _cubic_pooler(3))
    feat = d.features(_clip())
    assert feat.shape == (768,)
    assert np.allclose(feat, 6.0)


def test_three_frames_score_is_not_nan(monkeypatch):
    """Degenerate-but-accepted case: 3 frames passes the >=3 guard yet d2 has a single row,
    so the ddof=1 std over time would divide by (n-1)=0 and yield a silent NaN. The fix
    returns a defined value: a single sample has zero temporal spread, so the native score
    is 0 (negated to -0.0). We assert it is finite AND exactly 0, not NaN."""
    d = D3(device="cpu")
    _patch(monkeypatch, d, _cubic_pooler(3))
    s = d.score(_clip())
    assert np.isfinite(s)                                # no more silent NaN
    assert s == pytest.approx(0.0)                       # zero temporal spread -> 0 score


# ----------------------------------------------------------------------------- variant -> HF id

def test_variant_hf_ids():
    """The default and both named variants map to the right XCLIP HF repo ids."""
    assert D3().variant == "B/16"                        # paper default
    assert _HF["B/16"] == "microsoft/xclip-base-patch16"
    assert _HF["B/32"] == "microsoft/xclip-base-patch32"


@pytest.mark.parametrize("variant,hf_id", [
    ("B/16", "microsoft/xclip-base-patch16"),
    ("B/32", "microsoft/xclip-base-patch32"),
])
def test_load_resolves_variant_to_hf_id(monkeypatch, variant, hf_id):
    """_load() must hand the variant's HF id to XCLIPVisionModel.from_pretrained. We
    intercept from_pretrained (returning a dummy that supports .to(...).eval()) so no
    weights are fetched, and assert the id it was called with."""
    import transformers
    seen = {}

    class _Dummy:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    def _fake_from_pretrained(model_id, *a, **k):
        seen["id"] = model_id
        return _Dummy()

    monkeypatch.setattr(transformers.XCLIPVisionModel, "from_pretrained",
                        staticmethod(_fake_from_pretrained))
    d = D3(variant=variant, device="cpu")
    d._load()
    assert seen["id"] == hf_id


def test_unknown_variant_raises(monkeypatch):
    """An unknown variant has no HF id (_HF[self.variant]) -> KeyError before any download.
    from_pretrained is stubbed to fail loudly if it were ever reached."""
    import transformers

    def _boom(*a, **k):
        raise AssertionError("from_pretrained must not be called for an unknown variant")

    monkeypatch.setattr(transformers.XCLIPVisionModel, "from_pretrained",
                        staticmethod(_boom))
    d = D3(variant="B/14", device="cpu")
    with pytest.raises(KeyError):
        d._load()
