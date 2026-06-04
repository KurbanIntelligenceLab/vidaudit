"""STALL: the pure scoring math (whitening + Gaussian log-likelihood + percentile
calibration) and registration. The DINOv3 backbone is gated + heavy and is NOT exercised
here; it is validated on the cluster (like the NSG-VD / MLLM wrappers)."""
from __future__ import annotations

import numpy as np
import pytest

from vidaudit.detectors.stall import (_LOG2PI, gaussian_loglik, load_params,
                                       percentile_le, stall_scores, whiten)


def test_whiten():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = whiten(X, mu=np.array([1.0, 1.0]), W=np.eye(2) * 2.0)
    assert np.allclose(out, np.array([[0.0, 2.0], [4.0, 6.0]]))


def test_gaussian_loglik():
    d = 5
    assert np.isclose(gaussian_loglik(np.zeros((1, d)))[0], -0.5 * d * _LOG2PI)
    # larger norm -> strictly lower log-likelihood
    assert gaussian_loglik(np.array([[3.0, 4.0]]))[0] < gaussian_loglik(np.zeros((1, 2)))[0]


def test_percentile_le():
    calib = np.array([0.0, 1.0, 2.0, 3.0])
    assert percentile_le(-1.0, calib) == 0.0
    assert percentile_le(1.5, calib) == 0.5
    assert percentile_le(3.0, calib) == 1.0
    assert percentile_le(0.0, np.array([])) == 0.5      # empty calib -> neutral


def _ident_params(calib_n=200, d=8, seed=0):
    """Identity whitening + standard-normal real calibration (mu=0, W=I)."""
    rng = np.random.RandomState(seed)
    mu, W = np.zeros(d), np.eye(d)
    calib_spat = gaussian_loglik(whiten(rng.randn(calib_n, d), mu, W))
    diffs = rng.randn(calib_n, d)
    diffs /= np.linalg.norm(diffs, axis=1, keepdims=True)
    calib_temp = gaussian_loglik(whiten(diffs, mu, W))
    return {"mu_spat": mu, "W_spat": W, "mu_temp": mu, "W_temp": W,
            "calib_spat": calib_spat, "calib_temp": calib_temp}


def test_stall_scores_range_and_polarity():
    params = _ident_params()
    rng = np.random.RandomState(1)
    d = 8
    # real-like clip: frames ~ N(0, I), matching the calibration distribution
    sp, tp, sr, tr, final_real = stall_scores(rng.randn(6, d), params)
    assert 0.0 <= final_real <= 1.0 and 0.0 <= sp <= 1.0 and 0.0 <= tp <= 1.0
    # outlier clip: frames far from the real manifold -> lower spatial likelihood -> lower final
    _, _, _, _, final_out = stall_scores(rng.randn(6, d) + 12.0, params)
    assert final_out < final_real        # an outlier looks less real (higher p(generated))


def test_zero_diff_frames_ignored():
    """Identical consecutive frames (zero diff) must not crash the temporal branch."""
    params = _ident_params()
    emb = np.ones((4, 8)) * 0.3          # all frames identical -> all diffs zero
    sp, tp, sr, tr, final = stall_scores(emb, params)
    assert np.isinf(tr)                  # no valid transition -> +inf temporal raw
    assert 0.0 <= final <= 1.0


def test_load_params_roundtrip(tmp_path):
    p = _ident_params(calib_n=10, d=4)
    f = tmp_path / "stall_params.npz"
    np.savez(f, mu_spat=p["mu_spat"], W_spat=p["W_spat"], mu_temp=p["mu_temp"],
             W_temp=p["W_temp"], calib_spat=p["calib_spat"], calib_temp=p["calib_temp"])
    loaded = load_params(str(f))
    assert set(loaded) == {"mu_spat", "W_spat", "mu_temp", "W_temp", "calib_spat", "calib_temp"}
    assert np.allclose(loaded["W_spat"], np.eye(4))


def test_registered():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401
    assert "stall" in all_detectors()
    d = get("stall")
    assert d.has_native_head and d.has_features
    assert d.spec.needs_gpu and not d.spec.trainable
    with pytest.raises(ValueError):     # no params -> clear instruction, not a crash
        d.load_weights()
