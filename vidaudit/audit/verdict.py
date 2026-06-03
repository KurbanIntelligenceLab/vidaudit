"""The two orthogonal verdicts, mirroring the two figures in the audit paper.

`floor_verdict` asks whether LOGO-OOD clears the method's own real-vs-real (RvR)
floor: does the score reflect generation artifacts, or just which dataset the
reals came from? `deploy_tier` asks whether recall survives at a deployable
false-positive rate (TPR@0.1%). They are independent: a method can clear the
floor in AUC yet collapse at the operating point.

Both are derived from the audited numbers, so the leaderboard's labels are
reproducible from data rather than asserted.
"""
from __future__ import annotations

from typing import Optional

# --- floor thresholds (on LOGO-OOD vs RvR) ---
MARGIN_CERTIFIED = 0.15   # OOD - RvR at or above this clears the floor convincingly
MARGIN_MARGINAL = 0.10    # below this, the score barely exceeds the floor
RVR_CAUGHT = 0.70         # RvR at or above this means the score rides dataset identity

# --- deployability thresholds (on TPR@0.1%) ---
TPR_USABLE = 0.50
TPR_COLLAPSE = 0.05


def floor_verdict(logo_ood: Optional[float], rvr: Optional[float]) -> str:
    """certified | weak | marginal | caught | "" (not computable, e.g. <2 real sources)."""
    if logo_ood is None or rvr is None:
        return ""
    if rvr >= RVR_CAUGHT:
        return "caught"
    m = logo_ood - rvr
    if m < MARGIN_MARGINAL:
        return "marginal"
    if m >= MARGIN_CERTIFIED:
        return "certified"
    return "weak"


def deploy_tier(tpr01: Optional[float]) -> str:
    """usable | marginal | collapses | "" (operating point not computed)."""
    if tpr01 is None:
        return ""
    if tpr01 >= TPR_USABLE:
        return "usable"
    if tpr01 < TPR_COLLAPSE:
        return "collapses"
    return "marginal"
