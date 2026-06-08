"""Motion spectral features: the 13-d TemporalSpec extractor.

`mv.py` is vendored verbatim from the cluster TemporalSpec/features.py. It turns a
motion-vector array [T, H, W, 5] (channels SOURCE, DST_X, DST_Y, SRC_X, SRC_Y) into
Tier-0 statistics + Tier-1 per-region spectral features (slope, flatness, ACF, accel
moments). Used by the TemporalSpec (codec MV) and RAFT (optical-flow) detectors. The
Burg PSD path (frame counts < 32) needs the `spectrum` package.
"""
from __future__ import annotations

import numpy as np

from vidaudit.features.mv import extract_features  # noqa: F401

# The 13 features in canonical order (the matched-harness "MV 13-d" columns).
FEATURE_NAMES = [
    "mean_mv_mag", "mv_sparsity", "mv_variance", "mv_temporal_diff",
    "spectral_slope_median", "spectral_slope_iqr",
    "spectral_flatness_median", "spectral_flatness_iqr",
    "acf_decay_rate_median", "acf_decay_rate_iqr", "acf_first_zero_median",
    "accel_kurtosis_median", "accel_skewness_median",
]


def feature_vector(mv: np.ndarray, **kwargs):
    """extract_features -> ordered 13-d float array (FEATURE_NAMES), or None if the
    clip has too few motion frames."""
    d = extract_features(mv, **kwargs)
    if d is None:
        return None
    return np.array([d[k] for k in FEATURE_NAMES], dtype=float)
