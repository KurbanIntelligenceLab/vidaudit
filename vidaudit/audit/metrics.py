"""The audited metric tuple.

A high AUC does not imply a deployable detector, so the audit reports AUC plus the
operating-point and calibration metrics that decide whether the AUC is real:
partial AUC at a low FPR, true-positive rate at a fixed FPR, Brier, and ECE. Plus
the above-floor margin (vs the real-vs-real floor) that separates genuine
generalization from dataset-identity leakage.

Conventions: `y` is 1 for generated (the positive class), 0 for real. `s` is a
score where higher means more likely generated. `p` is a calibrated probability.
"""
from __future__ import annotations

import numpy as np


def auc(y, s) -> float:
    """ROC AUC (generated = positive)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, float)))


def pauc(y, s, max_fpr: float = 0.10) -> float:
    """Partial AUC over [0, max_fpr], McClish-standardized to [0.5, 1]."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, float), max_fpr=max_fpr))


def tpr_at_fpr(y, s, fpr: float = 0.001) -> float:
    """True-positive rate at a target false-positive rate (deployable recall).

    Highest TPR achievable while keeping FPR <= `fpr`, read off the ROC curve
    (matches the audited-metrics harness).
    """
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    f, t, _ = roc_curve(y, np.asarray(s, float))
    ok = f <= fpr
    return float(t[ok].max()) if ok.any() else 0.0


def brier(y, p) -> float:
    """Brier score (mean squared error of calibrated probabilities)."""
    y = np.asarray(y, float)
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins (10-bin)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = max(len(p), 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == 0 else (p > edges[i]) & (p <= edges[i + 1])
        if not m.any():
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def margin(logo_ood: float, rvr: float) -> float:
    """Above-floor margin: LOGO-OOD AUC minus the real-vs-real (RvR) floor."""
    return float(logo_ood - rvr)


def audited_tuple(y, s, p=None, fpr: float = 0.001) -> dict:
    """Compute the full audited metric tuple for one (scores, labels) split.

    `p` (calibrated probabilities) defaults to `s` for Brier/ECE if not given.
    """
    p = s if p is None else p
    return {
        "auc": auc(y, s),
        "pauc10": pauc(y, s, max_fpr=0.10),
        "tpr01": tpr_at_fpr(y, s, fpr=fpr),
        "tpr1": tpr_at_fpr(y, s, fpr=0.01),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }
