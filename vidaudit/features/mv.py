"""
Temporal spectral feature extraction from codec motion vectors.

Computes two tiers of features:
  Tier 0: Simple MV statistics (baselines to beat)
  Tier 1: Spectral features on a 4x4 spatial grid, aggregated via median/IQR

MV input shape: [T, H_mb, W_mb, 5] with channels [SOURCE, DST_X, DST_Y, SRC_X, SRC_Y].
"""

import warnings
import numpy as np
from scipy import signal, stats

try:
    import spectrum
    HAS_SPECTRUM = True
except ImportError:
    HAS_SPECTRUM = False

GRID_SIZE = 4
# Lowered from 8 to 7 to accommodate Text2Video-Zero (t2vz), whose clips have
# 8 raw frames with 7 non-I-frames after motion filtering. Other generators in
# the benchmark are unaffected (their non-I-frame medians are all >=12). Burg
# AR-order selection is the same in either regime (order = min(6, T // 3) = 2).
MIN_FRAMES = 7
BURG_ORDER = 6
NFFT_BURG = 128

# Channel indices matching MotionVectorExtractor/parse.py MVF class
CH_SOURCE = 0
CH_DST_X = 1
CH_DST_Y = 2
CH_SRC_X = 3
CH_SRC_Y = 4


def _compute_dx_dy(mv: np.ndarray):
    """Compute displacement vectors dx, dy from raw MV channels."""
    dx = mv[..., CH_DST_X] - mv[..., CH_SRC_X]
    dy = mv[..., CH_DST_Y] - mv[..., CH_SRC_Y]
    return dx.astype(np.float64), dy.astype(np.float64)


def _compute_magnitude(dx: np.ndarray, dy: np.ndarray):
    return np.sqrt(dx ** 2 + dy ** 2)


def _identify_iframes(mag: np.ndarray):
    """Return boolean mask: True for non-I-frames (frames with any motion)."""
    per_frame_sum = mag.reshape(mag.shape[0], -1).sum(axis=1)
    return per_frame_sum > 0


def _split_spatial_grid(mag: np.ndarray, grid_size: int = GRID_SIZE):
    """
    Split [T, H, W] magnitude into grid_size x grid_size spatial regions.
    Returns list of (T,) time series: mean magnitude per region.
    """
    T, H, W = mag.shape
    rh = max(1, H // grid_size)
    rw = max(1, W // grid_size)
    regions = []
    for gi in range(grid_size):
        for gj in range(grid_size):
            h_start = gi * rh
            h_end = H if gi == grid_size - 1 else (gi + 1) * rh
            w_start = gj * rw
            w_end = W if gj == grid_size - 1 else (gj + 1) * rw
            region = mag[:, h_start:h_end, w_start:w_end]
            regions.append(region.mean(axis=(1, 2)))
    return regions


def _estimate_psd(series: np.ndarray, burg_order: int = None,
                  nfft_burg: int = None):
    """
    Estimate PSD for a 1D time series.
    Uses Welch for T >= 32, Burg's method otherwise.
    Returns (freqs, psd) -- both 1D arrays.

    burg_order: override the Burg AR order cap (default: BURG_ORDER=6).
        Actual order is min(burg_order, T // 3) so the input length is
        always respected.
    nfft_burg: override the Burg NFFT (default: NFFT_BURG=128).
    """
    if burg_order is None:
        burg_order = BURG_ORDER
    if nfft_burg is None:
        nfft_burg = NFFT_BURG
    T = len(series)
    series = series - series.mean()
    if np.std(series) < 1e-12:
        return None, None

    if T >= 32:
        nperseg = min(T, max(8, T // 2))
        noverlap = nperseg // 2
        freqs, psd = signal.welch(series, fs=1.0, nperseg=nperseg,
                                  noverlap=noverlap, window='hann',
                                  detrend='constant')
        if len(freqs) < 3:
            return None, None
        return freqs, psd
    elif HAS_SPECTRUM:
        try:
            order = min(burg_order, T // 3)
            if order < 2:
                return None, None
            ar, _, _ = spectrum.arburg(series, order)
            psd_full = spectrum.arma2psd(ar, NFFT=nfft_burg)
            psd_full = np.abs(psd_full)
            psd_half = psd_full[:nfft_burg // 2 + 1]
            freqs = np.linspace(0, 0.5, len(psd_half))
            return freqs, psd_half
        except Exception:
            return None, None
    else:
        nperseg = min(T, max(4, T // 2))
        noverlap = nperseg // 2
        freqs, psd = signal.welch(series, fs=1.0, nperseg=nperseg,
                                  noverlap=noverlap, window='hann',
                                  detrend='constant')
        if len(freqs) < 3:
            return None, None
        return freqs, psd


def _spectral_slope(freqs: np.ndarray, psd: np.ndarray):
    """Log-log linear regression: S(f) = C * f^(-beta). Returns beta."""
    mask = freqs > 0
    if mask.sum() < 2:
        return np.nan
    log_f = np.log10(freqs[mask])
    psd_vals = psd[mask]
    psd_vals = np.maximum(psd_vals, 1e-30)
    log_p = np.log10(psd_vals)
    try:
        slope, _, _, _, _ = stats.linregress(log_f, log_p)
        return -slope
    except Exception:
        return np.nan


def _spectral_flatness(psd: np.ndarray):
    """Wiener entropy: exp(mean(log(PSD))) / mean(PSD)."""
    psd = psd[psd > 0]
    if len(psd) < 2:
        return np.nan
    log_mean = np.mean(np.log(psd))
    arith_mean = np.mean(psd)
    if arith_mean < 1e-30:
        return np.nan
    return np.exp(log_mean) / arith_mean


def _acf_features(series: np.ndarray, max_lag: int = None):
    """
    Compute ACF decay rate and first-zero crossing.
    decay_rate: fit R(tau) ~ exp(-tau/tau0), return tau0
    first_zero: first lag where ACF crosses zero

    max_lag: override the ACF cutoff (default: min(T-1, T//2)).
        Clipped to [1, T-1] so the input length is always respected.
    """
    T = len(series)
    series = series - series.mean()
    var = np.var(series)
    if var < 1e-12:
        return np.nan, np.nan

    if max_lag is None:
        max_lag = min(T - 1, T // 2)
    else:
        max_lag = max(1, min(max_lag, T - 1))
    acf = np.correlate(series, series, mode='full')
    acf = acf[T - 1:]
    acf = acf[:max_lag + 1] / (var * T)

    first_zero = np.nan
    for lag in range(1, len(acf)):
        if acf[lag] <= 0:
            first_zero = float(lag)
            break

    positive_lags = []
    positive_acf = []
    for lag in range(1, len(acf)):
        if acf[lag] > 0:
            positive_lags.append(lag)
            positive_acf.append(acf[lag])
    if len(positive_lags) >= 3:
        try:
            log_acf = np.log(np.array(positive_acf))
            lags_arr = np.array(positive_lags, dtype=float)
            slope, _, _, _, _ = stats.linregress(lags_arr, log_acf)
            if slope < 0:
                tau0 = -1.0 / slope
            else:
                tau0 = np.nan
        except Exception:
            tau0 = np.nan
    else:
        tau0 = np.nan

    return tau0, first_zero


def _acceleration_stats(series: np.ndarray):
    """Kurtosis and skewness of second-order temporal difference (acceleration)."""
    if len(series) < 3:
        return np.nan, np.nan
    accel = series[2:] - 2 * series[1:-1] + series[:-2]
    if np.std(accel) < 1e-12:
        return 0.0, 0.0
    return float(stats.kurtosis(accel, fisher=True)), float(stats.skew(accel))


def extract_features(mv: np.ndarray, target_frames: int = None,
                     burg_order: int = None, acf_max_lag: int = None):
    """
    Extract all spectral features from a single MV array.

    Args:
        mv: numpy array of shape [T, H_mb, W_mb, 5]
        target_frames: if set, require exactly this many non-I-frames after
            filtering (truncate to first K). Equalises the temporal window
            across generators to remove frame-count leakage -- the PSD method,
            ACF max-lag, and per-region time-series length all become uniform.
            Reject (return None) if fewer non-I-frames are available.
        burg_order: override the Burg AR-order cap for PSD estimation
            (default: BURG_ORDER=6). Effective order is min(burg_order, T//3).
        acf_max_lag: override the ACF cutoff (default: min(T-1, T//2)).

    Returns:
        dict with feature values and PSD vector, or None if video is too short.
    """
    dx, dy = _compute_dx_dy(mv)
    mag = _compute_magnitude(dx, dy)

    non_iframe_mask = _identify_iframes(mag)
    n_total = mag.shape[0]
    n_valid = int(non_iframe_mask.sum())

    threshold = target_frames if target_frames is not None else MIN_FRAMES
    if n_valid < threshold:
        return None

    mag = mag[non_iframe_mask]
    if target_frames is not None:
        mag = mag[:target_frames]
        n_valid = target_frames
    T, H, W = mag.shape

    # --- Tier 0: Global statistics ---
    global_mag = mag.mean(axis=(1, 2))
    mean_mv_mag = float(global_mag.mean())

    zero_frac_per_frame = (mag.reshape(T, -1) == 0).mean(axis=1)
    mv_sparsity = float(zero_frac_per_frame.mean())
    mv_variance = float(np.var(global_mag))
    mv_temporal_diff = float(np.mean(np.abs(np.diff(global_mag))))

    # --- Tier 1: Per-region spectral features ---
    regions = _split_spatial_grid(mag, GRID_SIZE)

    region_slopes = []
    region_flatness = []
    region_decay = []
    region_first_zero = []
    region_accel_kurt = []
    region_accel_skew = []
    region_psds = []

    for region_series in regions:
        freqs, psd = _estimate_psd(region_series, burg_order=burg_order)
        if freqs is None:
            continue

        region_slopes.append(_spectral_slope(freqs, psd))
        region_flatness.append(_spectral_flatness(psd))
        tau0, fz = _acf_features(region_series, max_lag=acf_max_lag)
        region_decay.append(tau0)
        region_first_zero.append(fz)
        kurt, skew = _acceleration_stats(region_series)
        region_accel_kurt.append(kurt)
        region_accel_skew.append(skew)
        region_psds.append(psd)

    if len(region_slopes) < 2:
        return None

    def _agg(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        if len(arr) == 0:
            return np.nan, np.nan
        return float(np.median(arr)), float(stats.iqr(arr))

    slope_med, slope_iqr = _agg(region_slopes)
    flat_med, flat_iqr = _agg(region_flatness)
    decay_med, decay_iqr = _agg(region_decay)
    # Degenerate ACF-decay branch: when NO region's autocorrelation admits an
    # exponential-decay fit (e.g. a periodic / fast-decorrelating signal whose ACF
    # crosses zero almost immediately, leaving < 3 positive lags), _agg sees an
    # all-NaN array and returns NaN, which would leak a silent NaN into the 13-d
    # vector. A failed fit means the series has no measurable correlation time, i.e.
    # it decorrelates immediately, so the decay rate is well-defined and finite at 0.0
    # (no temporal persistence). This matches the 0.0 convention _acceleration_stats
    # uses for "no temporal spread". Only this fully-degenerate case is rewritten;
    # the partial case (some regions fit) keeps _agg's NaN-dropping median/IQR intact,
    # so no non-degenerate output changes. Correct for raft too: a decorrelating
    # optical-flow clip is valid and should be scored, not rejected.
    if not np.isfinite(decay_med):
        decay_med, decay_iqr = 0.0, 0.0
    fz_med, _ = _agg(region_first_zero)
    kurt_med, _ = _agg(region_accel_kurt)
    skew_med, _ = _agg(region_accel_skew)

    # Spatially-averaged PSD: mean across valid regions, then store
    psd_lengths = [len(p) for p in region_psds]
    if psd_lengths:
        min_len = min(psd_lengths)
        trimmed = np.array([p[:min_len] for p in region_psds])
        avg_psd = trimmed.mean(axis=0)
    else:
        avg_psd = np.array([])

    return {
        # Tier 0
        'mean_mv_mag': mean_mv_mag,
        'mv_sparsity': mv_sparsity,
        'mv_variance': mv_variance,
        'mv_temporal_diff': mv_temporal_diff,
        # Tier 1
        'spectral_slope_median': slope_med,
        'spectral_slope_iqr': slope_iqr,
        'spectral_flatness_median': flat_med,
        'spectral_flatness_iqr': flat_iqr,
        'acf_decay_rate_median': decay_med,
        'acf_decay_rate_iqr': decay_iqr,
        'acf_first_zero_median': fz_med,
        'accel_kurtosis_median': kurt_med,
        'accel_skewness_median': skew_med,
        # Meta
        'n_frames_analyzed': int(n_valid),
        'n_frames_total': int(n_total),
        'n_frequency_bins': len(avg_psd),
        # PSD vector (for cross-codec analysis)
        'psd_vector': avg_psd,
    }
