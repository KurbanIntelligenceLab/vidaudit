"""NSG-VD (Zhang et al., NeurIPS 2025, arXiv:2510.08073): the pure NSG-velocity math and the
features() eval-path wiring, exercised with synthetic torch tensors and hand-computed values.
The ADM 256 diffusion score model and the Swin discriminator are heavy + gated and are NOT
loaded here (validated on the cluster, like the STALL / MLLM wrappers); we monkeypatch _load,
_preprocess, the score fn, and vidaudit.detectors._extract.decode_pil so the NSG arithmetic is
checked against exact known values without weights, GPU, network, or real video.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vidaudit.detectors.base import Clip


# ---- spec / capabilities --------------------------------------------------------

def test_registered():
    """nsgvd is registered as a features-only forensic detector with published (non-trainable,
    GPU) weights, a 300-d feature vector, and no native head (no score()), mirroring the wrapper."""
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "nsgvd" in all_detectors()
    d = get("nsgvd")
    assert d.has_features and not d.has_native_head
    assert d.spec.published_weights and not d.spec.trainable and d.spec.needs_gpu
    assert d.spec.family == "forensic"
    assert len(d.feature_names) == 300
    assert d.n_frames == 8 and d.size == 224 and d.diffuse_step == 5


# ---- _score_velocity: central diffs + endpoints + eps denom ---------------------

def _scalar_clip(values):
    """A (1, T, 1, 1, 1) tensor with one scalar per frame; everything reduces to per-frame
    arithmetic so pix / denom / velocity are exactly hand-computable."""
    return torch.tensor([float(v) for v in values]).view(1, len(values), 1, 1, 1)


def test_score_velocity_central_and_endpoint_diffs():
    """With score == 1 everywhere, velocity[t] == 1 / (pix[t] + eps), where pix uses
    central differences in the interior and forward/backward at the endpoints:
        T=4, image = [0, 1, 4, 9] ->
        pix[0] = image[1]-image[0]       = 1            (forward)
        pix[1] = (image[2]-image[0])/2   = 2            (central)
        pix[2] = (image[3]-image[1])/2   = 4            (central)
        pix[3] = image[3]-image[2]       = 5            (backward)
    so velocity ~ [1/1, 1/2, 1/4, 1/5] (eps=1e-10 is negligible here)."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu")
    image = _scalar_clip([0, 1, 4, 9])
    score = torch.ones_like(image)
    out = d._score_velocity(score, image)
    assert out.shape == (1, 4, 1, 1, 1)
    assert np.allclose(out.view(-1).numpy(), [1.0, 0.5, 0.25, 0.2], atol=1e-7)


def test_score_velocity_zero_score_is_zero():
    """score == 0 -> numerator zero, denom == 0 + eps == eps -> velocity is exactly 0 (the
    +eps guard keeps this 0/eps finite, not 0/0 NaN)."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu")
    image = _scalar_clip([0, 1, 4, 9])
    score = torch.zeros_like(image)
    out = d._score_velocity(score, image)
    assert np.array_equal(out.view(-1).numpy(), np.zeros(4))


def test_score_velocity_denominator_sums_over_channels():
    """The denom reduces over (C,H,W): denom[t] = sum(score[t]*pix[t]) + eps. With C=2 and a
    single forward-diff frame pair, pix[0] = image[1]-image[0] per channel; choosing
    image[0]=[0,0], image[1]=[1,2] and score[0]=[3,4] gives denom = 3*1 + 4*2 = 11, so
    velocity[0] = score / 11 = [3/11, 4/11] (eps negligible)."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu")
    # B=1, T=2, C=2, H=1, W=1
    image = torch.tensor([[[[[0.0]], [[0.0]]],
                           [[[1.0]], [[2.0]]]]])  # shape (1,2,2,1,1)
    assert image.shape == (1, 2, 2, 1, 1)
    score = torch.tensor([[[[[3.0]], [[4.0]]],
                           [[[3.0]], [[4.0]]]]])  # (1,2,2,1,1); frame-0 score = [3,4]
    out = d._score_velocity(score, image)
    # T=2: both frames are endpoints; pix[0]=image[1]-image[0]=[1,2]; pix[1]=image[1]-image[0]=[1,2]
    # denom[0] = 3*1 + 4*2 = 11 ; velocity[0] = [3/11, 4/11]
    assert np.allclose(out[0, 0].view(-1).numpy(), [3.0 / 11.0, 4.0 / 11.0], atol=1e-7)


def test_score_velocity_eps_guard_no_nan_on_identical_frames():
    """Degenerate input: identical consecutive frames -> all pixel diffs are 0, so denom == eps.
    The +eps guard keeps the result finite (a large 1/eps blowup) rather than producing NaN/inf;
    we assert the guard actually holds (no NaN, no inf), which is the property the +eps is for."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu")
    image = torch.full((1, 4, 1, 1, 1), 0.5)   # every frame identical -> all pix == 0
    score = torch.ones_like(image)
    out = d._score_velocity(score, image)       # eps=1e-10 default
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    # score/eps == 1/1e-10 == 1e10 in every frame (exact consequence of the +eps denom)
    assert np.allclose(out.view(-1).numpy(), np.full(4, 1e10), rtol=1e-6)


# ---- _compute_nsg: split first-half + 2x-1 rescale + curr_t + endpoint *100 -----

class _SplitScoreFn:
    """Stand-in for the ADM score predictor. Returns (B*T, 2C, H, W): the FIRST C channels are
    a constant `head` (the score _compute_nsg keeps via torch.split[0]); the trailing C channels
    are a sentinel that MUST be discarded. Records the rescaled input and timestep it was called
    with so the 2*x-1 rescale and curr_t = (step+1)/1000 wiring can be asserted."""

    def __init__(self, head=1.0, sentinel=999.0):
        self.head = head
        self.sentinel = sentinel
        self.seen_x = None
        self.seen_t = None

    def __call__(self, x, t):
        self.seen_x = x.clone()
        self.seen_t = t.clone()
        n, C, H, W = x.shape
        first = torch.full((n, C, H, W), float(self.head))
        second = torch.full((n, C, H, W), float(self.sentinel))
        return torch.cat([first, second], dim=1)   # (n, 2C, H, W)


def test_compute_nsg_split_rescale_timestep_and_endpoint_scaling():
    """End-to-end NSG math with the score fn stubbed to a constant first-half == 1:
      * torch.split keeps the FIRST C channels (sentinel 999 in the back half is discarded),
        so the effective score is 1 everywhere -> velocity = 1/(pix+eps);
      * for image=[0,1,4,9] (scalar frames) pix=[1,2,4,5] -> velocity=[1,1/2,1/4,1/5];
      * endpoints are multiplied by 100 -> nsg = [100, 0.5, 0.25, 20];
      * the score fn is fed 2*image-1 (the [-1,1] rescale) and curr_t=(5+1)/1000=0.006."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu", diffuse_step=5)
    fn = _SplitScoreFn(head=1.0, sentinel=999.0)
    d._score_fn = fn
    image = _scalar_clip([0, 1, 4, 9])          # (1,4,1,1,1)
    nsg = d._compute_nsg(image)
    assert nsg.shape == (1, 4, 1, 1, 1)
    assert np.allclose(nsg.view(-1).numpy(), [100.0, 0.5, 0.25, 20.0], atol=1e-6)
    # the rescale fed to the score fn is 2*image - 1
    assert np.allclose(fn.seen_x.view(-1).numpy(), [-1.0, 1.0, 7.0, 17.0])
    # timestep is the broadcast scalar (step+1)/1000 for all B*T rows
    assert fn.seen_t.shape == (4,)
    assert np.allclose(fn.seen_t.numpy(), np.full(4, 6.0 / 1000.0))


def test_compute_nsg_timestep_tracks_diffuse_step():
    """curr_t is exactly (diffuse_step + 1) / 1000; a different diffuse_step shifts it, confirming
    the constant is not hard-coded to 0.006."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu", diffuse_step=49)
    fn = _SplitScoreFn(head=1.0)
    d._score_fn = fn
    d._compute_nsg(_scalar_clip([0, 1, 4, 9]))
    assert np.allclose(fn.seen_t.numpy(), np.full(4, 50.0 / 1000.0))   # (49+1)/1000


# ---- features(): _preprocess + _score_fn + model.net(out_feature=True) ----------

class _StubNet:
    """model.net(nsg, out_feature=True) -> (logits, feat); feat is the (1, 300) per-clip vector
    that features() squeezes to a flat 300-d numpy float. Rows are 0..299 so the returned vector
    is exactly arange(300)."""

    def __init__(self):
        self.seen_nsg = None
        self.seen_out_feature = None

    def __call__(self, nsg, out_feature=False):
        self.seen_nsg = nsg
        self.seen_out_feature = out_feature
        feat = torch.arange(300, dtype=torch.float32).view(1, 300)
        return torch.zeros(1, 1), feat


class _StubModel:
    def __init__(self):
        self.net = _StubNet()


def _features_detector(monkeypatch, n_frames=3):
    """Build an NSGVD with the discriminator (_load), preprocessing (_preprocess), and ADM score
    fn all stubbed so features() runs the NSG -> net(out_feature=True) -> 300-d wiring with no
    weights. decode_pil is also patched so even an accidental real-decode path cannot fire."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu", n_frames=n_frames)
    model = _StubModel()
    d._model = model
    d._score_fn = _SplitScoreFn(head=1.0)
    monkeypatch.setattr(d, "_load", lambda: model)
    # synthetic preprocessed clip (1, T, 3, 224, 224) in [0,1] -> no real video decode
    dummy = torch.zeros(1, n_frames, 3, 224, 224)
    monkeypatch.setattr(d, "_preprocess", lambda clip: dummy)
    monkeypatch.setattr("vidaudit.detectors._extract.decode_pil",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("decode_pil must not be called in the unit test")))
    return d, model


def test_features_is_300d_float_vector(monkeypatch):
    """features() returns a flat 300-d float (np.float64) vector squeezed from net's (1,300) feat,
    and it asks net for the feature head (out_feature=True)."""
    d, model = _features_detector(monkeypatch)
    out = d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert isinstance(out, np.ndarray)
    assert out.shape == (300,)
    assert np.issubdtype(out.dtype, np.floating)          # features() ends in .astype(float)
    assert model.net.seen_out_feature is True


def test_features_values_match_net_feature_head(monkeypatch):
    """With the net stub returning feat == arange(300).view(1,300), features() returns exactly
    arange(300) (feat[0]), confirming the [0]-index + detach/cpu/numpy readout."""
    d, _ = _features_detector(monkeypatch)
    out = d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    assert np.array_equal(out, np.arange(300, dtype=float))


def test_features_passes_computed_nsg_to_net(monkeypatch):
    """The tensor handed to model.net is the (1, T, 3, 224, 224) NSG velocity, not the raw clip:
    with a constant-1 score head and an all-zero (identical-frame) input, every pixel diff is 0,
    so the +eps denom makes the NSG a finite 1e10 blowup (endpoints *100 -> 1e12). We assert the
    net saw a finite tensor of that exact shape (the NSG, not the zero input)."""
    d, model = _features_detector(monkeypatch, n_frames=3)
    d.features(Clip(video_id="v", path="/x.mp4", source="g", is_real=0))
    nsg = model.net.seen_nsg
    assert tuple(nsg.shape) == (1, 3, 3, 224, 224)
    assert not torch.isnan(nsg).any() and not torch.isinf(nsg).any()
    # zero input -> all pix==0 -> score(=1)/eps == 1e10 interior, *100 at the two endpoints
    interior = nsg[:, 1]
    endpoint = nsg[:, 0]
    assert np.allclose(interior.reshape(-1).numpy(), 1e10, rtol=1e-6)
    assert np.allclose(endpoint.reshape(-1).numpy(), 1e12, rtol=1e-6)


# ---- load_weights(): sets _disc_ckpt + nulls _model -----------------------------

def test_load_weights_sets_disc_ckpt_and_nulls_model():
    """load_weights(path) records the per-generator Swin discriminator checkpoint and invalidates
    any built model so the next _load() rebuilds against the new weights."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu")
    d._model = object()                       # pretend a model was already built
    d.load_weights("/weights/nsgvd_gen.pth")
    assert d._disc_ckpt == "/weights/nsgvd_gen.pth"
    assert d._model is None


def test_load_weights_none_keeps_existing_ckpt_but_nulls_model():
    """load_weights() with no path leaves the constructor-provided disc_ckpt in place (the zoo
    resolves it lazily at build time) but still nulls _model so it is rebuilt."""
    from vidaudit.detectors.nsgvd import NSGVD
    d = NSGVD(device="cpu", disc_ckpt="/ctor/disc.pth")
    d._model = object()
    d.load_weights()                          # path=None -> no overwrite
    assert d._disc_ckpt == "/ctor/disc.pth"
    assert d._model is None
