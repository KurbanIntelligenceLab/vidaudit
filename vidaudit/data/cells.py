"""Matched cells, feature-column resolution, and splits for the audited readout.

A *feature table* is a DataFrame with provenance columns (`video_id`, `generator`,
`is_real`, optionally `dataset`) plus arbitrary numeric feature columns. The audit
operates on this table: `apply_subset` restricts it to a matched cell (the same
clips evaluated across detectors), `resolve_feature_cols` selects the feature
columns, and `stratified_split` makes the fixed 80/20 splits.

Ported from the cluster harness (Baselines/evaluate_features.py) so numbers match.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

META_COLS = {
    "video_id", "generator", "label", "is_real",
    "mp4_path", "npy_path", "path", "pair", "split", "status", "dataset",
    "n_frames_analyzed", "n_frames_total", "n_frequency_bins",
}


def resolve_feature_cols(df: pd.DataFrame, spec: Union[str, Sequence[str], None]) -> List[str]:
    """Resolve feature columns from a spec.

    spec may be:
      * None or "auto:exclude=meta"  -> all numeric columns except META_COLS
      * "auto:prefix=v,a"            -> columns starting with a prefix + a digit
      * an explicit list / comma-separated string of column names
    """
    if spec is None or spec == "auto:exclude=meta":
        cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    elif isinstance(spec, str) and spec.startswith("auto:prefix="):
        prefixes = tuple(p for p in spec[len("auto:prefix="):].split(",") if p)
        cols = [c for c in df.columns
                if any(c.startswith(p) and c[len(p):len(p) + 1].isdigit() for p in prefixes)]
    else:
        cols = [c.strip() for c in spec.split(",")] if isinstance(spec, str) else [str(c) for c in spec]
        cols = [c for c in cols if c]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing from table: {missing}")
    if not cols:
        raise ValueError(f"no feature columns resolved from spec={spec!r}")
    return cols


def apply_subset(df: pd.DataFrame, subset: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """Restrict `df` to a matched cell defined by (video_id, generator) keys.

    `subset` is a CSV path or a DataFrame with (video_id, generator). Returns the
    filtered table and a per-generator extraction-failure audit (attempted vs present).
    """
    sub = pd.read_csv(subset, usecols=["video_id", "generator"]) if isinstance(subset, str) else subset.copy()
    sub["video_id"] = sub["video_id"].astype(str)
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)

    keys = set(zip(sub["video_id"], sub["generator"]))
    kept = pd.Series([(v, g) in keys for v, g in zip(df["video_id"], df["generator"])], index=df.index)
    df_filt = df[kept].reset_index(drop=True)

    per_gen = []
    for g, sub_g in sub.groupby("generator"):
        attempted = int(len(sub_g))
        succ = int((df_filt["generator"] == g).sum())
        per_gen.append({"generator": g, "attempted": attempted, "succeeded": succ,
                        "failed": attempted - succ, "failure_rate": (attempted - succ) / max(attempted, 1)})
    audit = {
        "subset_rows": int(len(sub)),
        "rows_in_subset": int(len(df_filt)),
        "rows_total": int(len(df)),
        "per_generator": sorted(per_gen, key=lambda r: r["generator"]),
    }
    return df_filt, audit


def stratified_split(frame: pd.DataFrame, strat: np.ndarray, seed: int = SEED):
    """Fixed 80/20 split stratified by `strat` (seed 42, matched across folds)."""
    return train_test_split(frame, test_size=0.2, stratify=strat, random_state=seed)
