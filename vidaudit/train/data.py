"""Feature-table data plumbing for the standard trainer.

The trainer learns a head over a *precomputed* per-clip feature table (the same
CSV the audit consumes), so training reuses the extract pipeline and never
re-decodes video in the loop. Preprocessing matches the audit readout exactly
(median-impute -> z-score), fit on train only and persisted with the checkpoint
so evaluation reproduces the transform. Label is generated-positive (y = 1 -
is_real), matching the audit's LOGO/RvR convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from vidaudit.data.cells import (apply_subset, read_feature_table,
                                 resolve_feature_cols, stratified_split)


@dataclass(slots=True)
class Preproc:
    """Fitted median-impute + z-score stats, persisted with the checkpoint."""
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), self.median, X)
        return (X - self.mean) / self.std


def _xy(df, feats: List[str]):
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()
    y = (1 - df["is_real"].astype(int)).to_numpy()          # generated-positive
    return X, y


@dataclass(slots=True)
class Loaders:
    train: DataLoader
    val: DataLoader
    in_dim: int
    feature_cols: List[str]
    preproc: Preproc


def make_loaders(cfg) -> Loaders:
    """Build train/val DataLoaders from the feature table named in `cfg`."""
    df = read_feature_table(cfg.features, arrow=cfg.arrow)
    if cfg.subset:
        df, _ = apply_subset(df, cfg.subset)
    feats = resolve_feature_cols(df, cfg.feature_cols)

    if cfg.val_features:
        tr_df, va_df = df, read_feature_table(cfg.val_features, arrow=cfg.arrow)
        if cfg.subset:
            va_df, _ = apply_subset(va_df, cfg.subset)
    else:
        y_all = (1 - df["is_real"].astype(int)).to_numpy()
        tr_df, va_df = stratified_split(df, y_all, seed=cfg.seed)

    Xtr, ytr = _xy(tr_df, feats)
    Xva, yva = _xy(va_df, feats)

    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    pre = Preproc(median=imp.statistics_.astype(np.float32),
                  mean=sc.mean_.astype(np.float32),
                  std=sc.scale_.astype(np.float32))

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(pre.transform(X).astype(np.float32)),
                           torch.from_numpy(y.astype(np.float32)))
        g = torch.Generator().manual_seed(cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, generator=g if shuffle else None,
                          drop_last=False)

    return Loaders(train=loader(Xtr, ytr, True), val=loader(Xva, yva, False),
                   in_dim=len(feats), feature_cols=feats, preproc=pre)
