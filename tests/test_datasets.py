"""Dataset adapters: native-layout enumeration (GenVidBench dirs, AIGVDBench zips) +
end-to-end prepare_dataset (P1 re-encode + P2 filter) on tiny synthetic downloads."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile

import pandas as pd
import pytest

from vidaudit.data.datasets import all_datasets, get_dataset


def test_registry_has_builtin_datasets():
    assert {"genvidbench", "aigvdbench"} <= set(all_datasets())


def test_genvidbench_scan_maps_sources_and_realness(tmp_path):
    root = tmp_path / "gvb"
    for pair, src in [("Pair2", "cogvideo"), ("Pair2", "hd_vg_130m"), ("Pair1", "vript")]:
        d = root / pair / src / src                      # GenVidBench double-nesting
        d.mkdir(parents=True)
        (d / "0001___a prompt.mp4").write_bytes(b"")      # scan only reads names
    clips = {c.source: c for c in get_dataset("genvidbench").scan(str(root))}
    assert set(clips) == {"cogvideo", "hd_vg_130m", "vript"}
    assert clips["cogvideo"].is_real == 0                 # generator
    assert clips["hd_vg_130m"].is_real == 1 and clips["vript"].is_real == 1   # real sources
    assert clips["cogvideo"].video_id == "0001___a prompt" and clips["cogvideo"].dataset == "genvidbench"


def test_genvidbench_scan_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(get_dataset("genvidbench").scan(str(tmp_path)))


def test_aigvdbench_scan_reads_zip_members(tmp_path):
    base = tmp_path / "aigvd" / "AIGVDBench"
    (base / "Real").mkdir(parents=True)
    (base / "OpenSource" / "T2V").mkdir(parents=True)
    with zipfile.ZipFile(base / "Real" / "Real.zip", "w") as z:
        z.writestr("clip_a.mp4", b"x")
    with zipfile.ZipFile(base / "OpenSource" / "T2V" / "Wan2.1.zip", "w") as z:
        z.writestr("vids/clip_b.mp4", b"x")
    clips = {c.source: c for c in get_dataset("aigvdbench").scan(str(tmp_path / "aigvd"))}
    assert clips["Real"].is_real == 1 and clips["Real"].video_id == "clip_a"
    assert clips["Wan2.1"].is_real == 0 and clips["Wan2.1"].video_id == "clip_b"
    assert clips["Wan2.1"].meta["member"].endswith("clip_b.mp4")        # zip-backed
    # one real source -> RvR floor auto-disables
    assert get_dataset("aigvdbench").spec.supports_rvr is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_prepare_dataset_end_to_end(tmp_path):
    from vidaudit.data.fetch import prepare_dataset
    from vidaudit.data.filters import KFilter
    root = tmp_path / "gvb"
    for src in ["cogvideo", "hd_vg_130m"]:
        d = root / "Pair2" / src / src
        d.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=8:duration=2",
                        "-pix_fmt", "yuv420p", str(d / "0001___p.mp4")], check=True, capture_output=True)
    out = tmp_path / "out"
    man = prepare_dataset("genvidbench", str(root), str(out), kfilter=KFilter(min_frames=8))
    df = pd.read_csv(man)
    assert len(df) == 2 and set(df["generator"]) == {"cogvideo", "hd_vg_130m"}
    assert df.loc[df.generator == "hd_vg_130m", "is_real"].iloc[0] == 1
    assert all(os.path.exists(p) for p in df["mp4_path"])
    prov = json.load(open(out / "provenance.json"))
    assert prov["dataset"] == "genvidbench" and prov["homepage"] and prov["recipe"].startswith("h264-")
