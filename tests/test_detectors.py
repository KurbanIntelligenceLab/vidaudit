"""Tests for the model-zoo plumbing: registry + specs, the extraction harness, and the
weight-fetch zoo. These exercise the wrapper/harness logic without loading any heavy
backbone (a stub detector stands in), so they are fast and need no weights or GPU.
"""
import numpy as np
import pandas as pd
import pytest

from vidaudit.detectors.base import Clip, Detector, DetectorSpec

EXPECTED = {"d3", "restrav", "clip", "raft", "temporalspec", "waverep", "fvmd", "nsgvd"}


def test_all_eight_detectors_register_with_valid_specs():
    from vidaudit.detectors import all_detectors, get
    names = set(all_detectors())
    assert EXPECTED <= names, f"missing: {EXPECTED - names}"
    for n in sorted(EXPECTED):
        det = get(n)                       # constructs the wrapper (no model load)
        s = det.spec
        assert s.name and s.family and s.paper, f"{n}: incomplete spec"
        assert det.has_features or det.has_native_head, f"{n}: no evidence interface"
        fn = getattr(det, "feature_names", None)
        if fn is not None:
            assert isinstance(fn, list) and len(fn) > 0


def test_clips_from_manifest(tmp_path):
    from vidaudit.detectors._extract import clips_from_manifest
    csv = tmp_path / "m.csv"
    pd.DataFrame({"video_id": ["a", "b"], "generator": ["svd", "vript"],
                  "label": ["fake", "real"], "is_real": [0, 1],
                  "mp4_path": ["/x/a.mp4", "/x/b.mp4"]}).to_csv(csv, index=False)
    clips = clips_from_manifest(csv)
    assert len(clips) == 2
    assert clips[0].video_id == "a" and clips[0].source == "svd" and clips[0].is_real == 0
    assert clips[1].is_real == 1 and clips[0].meta["label"] == "fake"


class _Stub(Detector):
    spec = DetectorSpec(name="Stub", family="x", paper="none")
    feature_names = ["f0", "f1", "f2"]

    def features(self, clip):
        return np.array([1.0, 2.0, 3.0])


def _clips(n=3):
    return [Clip(video_id=f"v{i}", path="/x.mp4", source=("svd" if i % 2 else "real"),
                 is_real=i % 2, meta={"label": "l"}) for i in range(n)]


def test_extract_table_schema_and_named_columns(tmp_path):
    from vidaudit.detectors._extract import extract_table
    out = tmp_path / "feat.csv"
    df = extract_table(_Stub(), _clips(3), out=str(out), verbose=False)
    assert len(df) == 3
    assert {"video_id", "generator", "label", "is_real", "f0", "f1", "f2"} <= set(df.columns)
    assert out.exists()


def test_extract_table_resume_skips_done(tmp_path):
    from vidaudit.detectors._extract import extract_table
    out = tmp_path / "feat.csv"
    extract_table(_Stub(), _clips(3), out=str(out), verbose=False)
    df2 = extract_table(_Stub(), _clips(3), out=str(out), verbose=False)  # all already done
    assert len(df2) == 3 and df2["video_id"].nunique() == 3  # no duplicates


def test_extract_table_default_prefix_columns(tmp_path):
    from vidaudit.detectors._extract import extract_table

    class _NoNames(Detector):
        spec = DetectorSpec(name="MyDet", family="x", paper="none")
        def features(self, clip):
            return np.zeros(4)

    df = extract_table(_NoNames(), _clips(1), verbose=False)
    assert {"mydet_0", "mydet_1", "mydet_2", "mydet_3"} <= set(df.columns)


def test_zoo_manifest_entries_and_fetch_error():
    from vidaudit import zoo
    m = zoo.load_manifest()
    assert {"waverep", "fvmd", "nsgvd", "nsgvd-adm"} <= set(m.keys())
    assert m["waverep"]["weights"][0]["sha256"]            # sha256 recorded
    # WaveRep's url is a cluster-storage note, not HTTP -> a clear error, no download attempt.
    with pytest.raises(RuntimeError):
        zoo.fetch_weights("waverep")
