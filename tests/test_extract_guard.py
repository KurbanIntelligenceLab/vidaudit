"""extract_table must never write a non-finite (NaN/inf) feature or score row: a degenerate
clip is counted as a failure and skipped, so the matched-harness readout is never poisoned.
This is the centralized defense-in-depth guard backing the per-detector degenerate fixes."""
from __future__ import annotations

import numpy as np

from vidaudit.detectors._extract import extract_table
from vidaudit.detectors.base import Clip, Detector, DetectorSpec


class _NanScore(Detector):
    spec = DetectorSpec(name="nanscore")

    def score(self, clip):
        return float("nan")


class _NanFeat(Detector):
    spec = DetectorSpec(name="nanfeat")

    def features(self, clip):
        return np.array([1.0, np.nan, 3.0])      # one NaN entry must drop the whole row


class _GoodScore(Detector):
    spec = DetectorSpec(name="good")

    def score(self, clip):
        return 0.25 if clip.is_real else 0.9


def _clips(n=3):
    return [Clip(video_id=f"v{i}", path=f"/x/{i}.mp4", source="g", is_real=i % 2)
            for i in range(n)]


def test_nan_score_rows_are_skipped(tmp_path):
    df = extract_table(_NanScore(), _clips(3), kind="score",
                       out=str(tmp_path / "s.csv"), verbose=False)
    assert len(df) == 0                          # all NaN scores dropped as failures


def test_nan_feature_rows_are_skipped(tmp_path):
    df = extract_table(_NanFeat(), _clips(2), kind="features",
                       out=str(tmp_path / "f.csv"), verbose=False)
    assert len(df) == 0                          # a NaN anywhere in the vector drops the row


def test_finite_rows_are_kept(tmp_path):
    df = extract_table(_GoodScore(), _clips(4), kind="score",
                       out=str(tmp_path / "g.csv"), verbose=False)
    assert len(df) == 4
    assert np.all(np.isfinite(df["score"].to_numpy()))
