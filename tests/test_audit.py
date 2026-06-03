"""Mechanics tests for the audit engine on synthetic feature tables.

These verify the readout/LOGO/RvR/verdict wiring behaves correctly; reproducing
the paper's exact numbers is a separate check against the real feature CSVs.
"""
import numpy as np
import pandas as pd
import pytest

from vidaudit.audit import metrics as M
from vidaudit.audit import verdict as V
from vidaudit.audit.protocol import audit_features
from vidaudit.data.cells import apply_subset, resolve_feature_cols


def make_table(rng, *, gens, n_real=150, n_gen=90, gen_mu=0.0, dim=8, src_shift=(0.0, 0.0)):
    """Two real sources + several generators, `dim` gaussian features."""
    rows = []
    k = 0
    for si, src in enumerate(("src_a", "src_b")):
        X = rng.normal(src_shift[si], 1.0, size=(n_real, dim))
        for i in range(n_real):
            rows.append({"video_id": f"r{k}", "generator": src, "is_real": 1,
                         **{f"f{j}": X[i, j] for j in range(dim)}})
            k += 1
    for g in gens:
        X = rng.normal(gen_mu, 1.0, size=(n_gen, dim))
        for i in range(n_gen):
            rows.append({"video_id": f"v{k}", "generator": g, "is_real": 0,
                         **{f"f{j}": X[i, j] for j in range(dim)}})
            k += 1
    return pd.DataFrame(rows)


def test_certified_when_separable_and_no_source_bias():
    rng = np.random.default_rng(0)
    df = make_table(rng, gens=["g1", "g2", "g3", "g4"], gen_mu=1.7)  # gens shifted; reals both N(0,1)
    rec = audit_features(df)
    assert rec["logo_ood"] > 0.80          # generalizes to held-out generators
    assert rec["rvr"] < 0.65               # reals not separable -> low floor
    assert rec["margin"] > 0.15
    assert rec["floor_verdict"] == "certified"
    assert rec["deploy_tier"] in ("usable", "marginal")  # not collapsed


def test_caught_when_features_encode_dataset_identity():
    rng = np.random.default_rng(1)
    df = make_table(rng, gens=["g1", "g2", "g3"], gen_mu=2.0, src_shift=(0.0, 4.0))  # real sources separable
    rec = audit_features(df)
    assert rec["rvr"] > 0.70               # real-vs-real high -> dataset identity
    assert rec["floor_verdict"] == "caught"


def test_marginal_on_noise():
    rng = np.random.default_rng(2)
    df = make_table(rng, gens=["g1", "g2", "g3"], gen_mu=0.0)  # no real-vs-gen signal
    rec = audit_features(df)
    assert rec["logo_ood"] < 0.65
    assert rec["floor_verdict"] in ("marginal", "weak")


def test_rvr_disabled_with_one_real_source():
    rng = np.random.default_rng(3)
    df = make_table(rng, gens=["g1", "g2"], gen_mu=1.5)
    df = df[df["generator"] != "src_b"].reset_index(drop=True)  # drop to a single real source
    rec = audit_features(df)
    assert rec["rvr"] is None and rec["margin"] is None


def test_metrics_basic():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert M.auc(y, s_perfect) == 1.0
    assert M.tpr_at_fpr(y, s_perfect, 0.001) == 1.0      # separable -> full recall at FPR 0
    assert M.pauc(y, s_perfect, max_fpr=0.10) == 1.0
    assert M.tpr_at_fpr(y, np.full(8, 0.5), 0.01) == 0.0  # no signal


def test_verdict_thresholds():
    assert V.floor_verdict(0.996, 0.534) == "certified"
    assert V.floor_verdict(0.852, 0.766) == "caught"     # high RvR dominates
    assert V.floor_verdict(0.660, 0.596) == "marginal"
    assert V.floor_verdict(None, None) == ""
    assert V.deploy_tier(0.816) == "usable"
    assert V.deploy_tier(0.024) == "collapses"
    assert V.deploy_tier(0.30) == "marginal"


def test_cells_resolve_and_subset():
    df = pd.DataFrame({"video_id": ["a", "b", "c"], "generator": ["g1", "g1", "g2"],
                       "is_real": [0, 0, 0], "f0": [1.0, 2.0, 3.0], "f1": [4.0, 5.0, 6.0]})
    assert resolve_feature_cols(df, None) == ["f0", "f1"]            # excludes meta cols
    assert resolve_feature_cols(df, "f0") == ["f0"]
    with pytest.raises(ValueError):
        resolve_feature_cols(df, "nope")
    sub = pd.DataFrame({"video_id": ["a", "c"], "generator": ["g1", "g2"]})
    filt, audit = apply_subset(df, sub)
    assert len(filt) == 2 and audit["rows_in_subset"] == 2
