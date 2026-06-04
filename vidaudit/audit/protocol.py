"""The audited evaluation engine: run a feature table through the matched-harness
readout under leave-one-generator-out (LOGO), the real-vs-real floor, and the
audited metric tuple, then assign the two verdicts. One call per detector ->
one leaderboard row.

Readout (ported from the cluster harness so numbers reproduce): median-impute ->
z-score -> optional reducer -> balanced L2-LR (lbfgs, C=1, seed 42). For each
held-out generator the OOD readout trains on real + the *other* generators and
scores a real+held test set; per-generator OOD predictions feed the operating-point
metrics (pAUC@10%, TPR@FPR), and the OOD AUC mean is the headline LOGO number.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vidaudit.audit import metrics as M
from vidaudit.audit import verdict as V
from vidaudit.data.cells import SEED, apply_subset, resolve_feature_cols

INNER_C_GRID = [0.01, 0.1, 1.0, 10.0]


def build_readout(n_features: int, *, reducer: str = "none", n_components: int = 13,
                  solver: str = "lbfgs", inner_cv: bool = False, classifier: str = "lr"):
    """median-impute -> z-score -> optional reducer -> classifier (default balanced L2-LR)."""
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if reducer == "pca":
        from sklearn.decomposition import PCA
        steps.append(("reduce", PCA(n_components=min(n_components, n_features), random_state=SEED)))
    elif reducer == "topk_l2":
        from sklearn.feature_selection import SelectFromModel
        ranker = LogisticRegression(penalty="l2", C=1.0, solver=solver, max_iter=2000,
                                    class_weight="balanced", random_state=SEED)
        steps.append(("reduce", SelectFromModel(ranker, max_features=min(n_components, n_features),
                                                threshold=-np.inf)))
    elif reducer != "none":
        raise ValueError(f"unknown reducer: {reducer}")

    if classifier == "xgboost":
        from xgboost import XGBClassifier
        steps.append(("clf", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                           random_state=SEED, n_jobs=-1, tree_method="hist")))
        return Pipeline(steps)

    if classifier == "lightgbm":
        from lightgbm import LGBMClassifier
        steps.append(("clf", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                            random_state=SEED, n_jobs=-1, verbose=-1)))
        return Pipeline(steps)

    steps.append(("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                            solver=solver, random_state=SEED)))
    pipe = Pipeline(steps)
    if inner_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(pipe, {"clf__C": INNER_C_GRID}, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    return pipe


def _num(frame: pd.DataFrame, feats: Sequence[str]) -> pd.DataFrame:
    # coerce to float64 so the readout is identical regardless of the read backend
    # (numpy vs pandas pyarrow dtypes for wide feature tables)
    return frame[list(feats)].apply(pd.to_numeric, errors="coerce").astype("float64")


def run_logo(df: pd.DataFrame, feats: Sequence[str], **opts) -> Dict:
    """Per-generator ID and OOD AUC + the OOD test predictions for the metric suite."""
    real = df[df["is_real"] == 1].reset_index(drop=True)
    # Canonical audited split: non-stratified ShuffleSplit at SEED (matches the cluster
    # run_audited_metrics harness). NB: passing stratify=<constant> would route through
    # StratifiedShuffleSplit and pick a *disjoint* seed-SEED partition (the source of the
    # FVMD 0.894-vs-0.880 difference vs evaluate_features); that gap is split noise,
    # bounded by the bootstrap CI, not a methodology change. Keep this non-stratified.
    real_tr, real_te = train_test_split(real, test_size=0.2, random_state=SEED)
    gens = sorted(df.loc[df["is_real"] == 0, "generator"].astype(str).unique().tolist())
    gs = {g: train_test_split(df[df["generator"].astype(str) == g].reset_index(drop=True),
                              test_size=0.2, random_state=SEED) for g in gens}

    per_gen: Dict[str, Dict] = {}
    ood_preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for held in gens:
        held_tr, held_te = gs[held]
        others_tr = [gs[g][0] for g in gens if g != held]

        Xte = _num(pd.concat([real_te, held_te], ignore_index=True), feats)
        yte = np.concatenate([np.zeros(len(real_te), int), np.ones(len(held_te), int)])

        X_ood = _num(pd.concat([real_tr, *others_tr], ignore_index=True), feats)
        y_ood = np.concatenate([np.zeros(len(real_tr), int)] + [np.ones(len(o), int) for o in others_tr])
        X_id = _num(pd.concat([real_tr, held_tr], ignore_index=True), feats)
        y_id = np.concatenate([np.zeros(len(real_tr), int), np.ones(len(held_tr), int)])

        pipe_ood = build_readout(len(feats), **opts).fit(X_ood, y_ood)
        pipe_id = build_readout(len(feats), **opts).fit(X_id, y_id)
        s_ood = pipe_ood.predict_proba(Xte)[:, 1]
        s_id = pipe_id.predict_proba(Xte)[:, 1]

        ood_preds[held] = (yte, s_ood)
        per_gen[held] = {
            "ID_auc": float(roc_auc_score(yte, s_id)),
            "OOD_auc": float(roc_auc_score(yte, s_ood)),
            "n_real_test": int((yte == 0).sum()),
            "n_held_test": int((yte == 1).sum()),
        }
    logo_id = float(np.mean([v["ID_auc"] for v in per_gen.values()]))
    logo_ood = float(np.mean([v["OOD_auc"] for v in per_gen.values()]))
    return {"per_generator": per_gen, "ood_preds": ood_preds, "logo_id": logo_id, "logo_ood": logo_ood}


def run_rvr(df: pd.DataFrame, feats: Sequence[str], **opts) -> Optional[Dict]:
    """Real-vs-real floor: classify the two real sources. None if <2 real sources."""
    real = df[df["is_real"] == 1].copy()
    sources = sorted(real["generator"].astype(str).unique().tolist())
    if len(sources) < 2:
        return None
    y = (real["generator"].astype(str) == sources[-1]).astype(int).to_numpy()
    X = _num(real, feats)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = build_readout(len(feats), **opts).fit(X_tr, y_tr)
    return {"sources": sources, "auc": float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))}


def audited_suite(ood_preds: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """Mean over held-out generators of the audited metric tuple."""
    rows = {k: [] for k in ("auc", "pauc10", "tpr1", "tpr01", "brier", "ece")}
    for y, s in ood_preds.values():
        rows["auc"].append(M.auc(y, s))
        rows["pauc10"].append(M.pauc(y, s, max_fpr=0.10))
        rows["tpr1"].append(M.tpr_at_fpr(y, s, 0.01))
        rows["tpr01"].append(M.tpr_at_fpr(y, s, 0.001))
        rows["brier"].append(M.brier(y, s))
        rows["ece"].append(M.ece(y, s))
    return {k: float(np.mean(v)) for k, v in rows.items()}


def audit_features(df: pd.DataFrame, feature_cols: Union[str, Sequence[str], None] = None, *,
                   subset: Union[str, pd.DataFrame, None] = None, reducer: str = "none",
                   n_components: int = 13, solver: str = "lbfgs", inner_cv: bool = False,
                   classifier: str = "lr") -> Dict:
    """Run the full audit on a feature table and return one leaderboard record."""
    failure = None
    if subset is not None:
        df, failure = apply_subset(df, subset)
    feats = resolve_feature_cols(df, feature_cols)

    block = _num(df, feats)
    keep = block.notna().any(axis=1)
    df = df[keep].reset_index(drop=True)

    opts = dict(reducer=reducer, n_components=n_components, solver=solver,
                inner_cv=inner_cv, classifier=classifier)
    logo = run_logo(df, feats, **opts)
    suite = audited_suite(logo["ood_preds"])
    rvr = run_rvr(df, feats, **opts)
    rvr_auc = rvr["auc"] if rvr else None
    margin = None if rvr_auc is None else logo["logo_ood"] - rvr_auc

    return {
        "n_features": len(feats),
        "n_clips": int(len(df)),
        "logo_id": round(logo["logo_id"], 4),
        "logo_ood": round(logo["logo_ood"], 4),
        "rvr": None if rvr_auc is None else round(rvr_auc, 4),
        "margin": None if margin is None else round(margin, 4),
        "pauc10": round(suite["pauc10"], 4),
        "tpr1": round(suite["tpr1"], 4),
        "tpr01": round(suite["tpr01"], 4),
        "brier": round(suite["brier"], 4),
        "ece": round(suite["ece"], 4),
        "floor_verdict": V.floor_verdict(logo["logo_ood"], rvr_auc),
        "deploy_tier": V.deploy_tier(suite["tpr01"]),
        "per_generator": logo["per_generator"],
        "rvr_sources": rvr["sources"] if rvr else None,
        "failure_audit": failure,
    }


# --- native-head / trained-head path: audit a FIXED per-clip score (no readout) ------
def run_logo_scores(df: pd.DataFrame, score_col: str) -> Dict:
    """Per-held-generator OOD AUC of a FIXED per-clip score (a native or trained head),
    on the SAME test split run_logo uses, so it is directly comparable to the readout
    column. No retraining, so for a fixed score ID == OOD."""
    real = df[df["is_real"] == 1].reset_index(drop=True)
    _, real_te = train_test_split(real, test_size=0.2, random_state=SEED)
    gens = sorted(df.loc[df["is_real"] == 0, "generator"].astype(str).unique().tolist())
    gs = {g: train_test_split(df[df["generator"].astype(str) == g].reset_index(drop=True),
                              test_size=0.2, random_state=SEED) for g in gens}
    ood_preds, per_gen = {}, {}
    for held in gens:
        _, held_te = gs[held]
        te = pd.concat([real_te, held_te], ignore_index=True)
        yte = np.concatenate([np.zeros(len(real_te), int), np.ones(len(held_te), int)])
        s = pd.to_numeric(te[score_col], errors="coerce").to_numpy()
        ood_preds[held] = (yte, s)
        a = float(M.auc(yte, s))
        per_gen[held] = {"ID_auc": a, "OOD_auc": a,
                         "n_real_test": int((yte == 0).sum()), "n_held_test": int((yte == 1).sum())}
    logo = float(np.mean([v["OOD_auc"] for v in per_gen.values()])) if per_gen else float("nan")
    return {"per_generator": per_gen, "ood_preds": ood_preds, "logo_id": logo, "logo_ood": logo}


def run_rvr_scores(df: pd.DataFrame, score_col: str) -> Optional[Dict]:
    """Real-vs-real floor for a fixed score: does the score separate the two real
    sources (dataset-identity leakage)? Folded AUC (direction-agnostic, since the score
    was not fit to this task) on the same stratified real test split. None if <2 reals."""
    real = df[df["is_real"] == 1].reset_index(drop=True)
    sources = sorted(real["generator"].astype(str).unique().tolist())
    if len(sources) < 2:
        return None
    y = (real["generator"].astype(str) == sources[-1]).astype(int).to_numpy()
    s = pd.to_numeric(real[score_col], errors="coerce").to_numpy()
    _, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    a = float(M.auc(y[te], s[te]))
    return {"sources": sources, "auc": max(a, 1.0 - a)}


def audit_scores(df: pd.DataFrame, score_col: str = "score", *,
                 subset: Union[str, pd.DataFrame, None] = None) -> Dict:
    """Audit a FIXED per-clip score (native head like D3's, or a trained head's output)
    through the same structure as audit_features but WITHOUT retraining a readout. This
    closes the loop: `extract --kind score` (or a trained head's predictions) feed
    straight into the audited metric tuple + both verdicts."""
    failure = None
    if subset is not None:
        df, failure = apply_subset(df, subset)
    if score_col not in df.columns:
        raise ValueError(f"score column {score_col!r} not in table; columns: {list(df.columns)[:12]}")
    df = df[pd.to_numeric(df[score_col], errors="coerce").notna()].reset_index(drop=True)

    logo = run_logo_scores(df, score_col)
    suite = audited_suite(logo["ood_preds"])
    rvr = run_rvr_scores(df, score_col)
    rvr_auc = rvr["auc"] if rvr else None
    margin = None if rvr_auc is None else logo["logo_ood"] - rvr_auc
    return {
        "readout": "native-score", "score_col": score_col,
        "n_features": 0, "n_clips": int(len(df)),
        "logo_id": round(logo["logo_id"], 4), "logo_ood": round(logo["logo_ood"], 4),
        "rvr": None if rvr_auc is None else round(rvr_auc, 4),
        "margin": None if margin is None else round(margin, 4),
        "pauc10": round(suite["pauc10"], 4), "tpr1": round(suite["tpr1"], 4),
        "tpr01": round(suite["tpr01"], 4), "brier": round(suite["brier"], 4),
        "ece": round(suite["ece"], 4),
        "floor_verdict": V.floor_verdict(logo["logo_ood"], rvr_auc),
        "deploy_tier": V.deploy_tier(suite["tpr01"]),
        "per_generator": logo["per_generator"], "rvr_sources": rvr["sources"] if rvr else None,
        "failure_audit": failure,
    }
