"""Data package: P1 canonical recipe, P2 length filter, Croissant metadata, and an
ffmpeg-guarded end-to-end reconstruct on tiny synthesized clips."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pandas as pd
import pytest

from vidaudit.data import (CANONICAL, CanonicalSpec, KFilter, byo_instructions,
                           download_file, feature_table_croissant, ffmpeg_cmd,
                           frame_count, recipe_id, reconstruct)


# ---- P1 canonical recipe ---------------------------------------------------
def test_canonical_cmd_and_recipe_id():
    cmd = ffmpeg_cmd("in.mp4", "out.mp4", CANONICAL)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4"
    assert "libx264" in cmd and "yuv420p" in cmd and "-an" in cmd      # H.264, fixed pix_fmt, no audio
    assert "23" in cmd                                                 # the fixed CRF
    rid = recipe_id(CANONICAL)
    assert rid.startswith("h264-") and rid == recipe_id(CANONICAL)     # stable
    assert recipe_id(CanonicalSpec(crf=18)) != rid                     # changes with the recipe


# ---- P2 length filter (inject counts, no video needed) ---------------------
def test_kfilter_drops_short_and_reports_leakage():
    df = pd.DataFrame([
        {"video_id": f"r{i}", "generator": "real_a" if i < 2 else "real_b", "is_real": 1}
        for i in range(4)
    ] + [
        {"video_id": f"g{i}", "generator": ["g1", "g2", "g3"][i % 3], "is_real": 0}
        for i in range(6)
    ])
    counts = pd.Series([8, 20, 30, 40,            # real: one below min
                        12, 25, 50, 60, 18, 70])  # gen: one below min
    kept, rep = KFilter(min_frames=16).apply(df, counts=counts)

    assert rep["n_in"] == 10 and rep["n_dropped"] == 2 and rep["n_kept"] == 8
    assert len(kept) == 8
    assert 0.5 <= rep["length_leakage_before"] <= 1.0                  # length leaks pre-filter
    assert 0.5 <= rep["length_leakage_after"] <= 1.0                   # both classes survive -> defined
    assert {r["generator"] for r in rep["per_generator"]} >= {"g1", "g2", "g3"}


# ---- Croissant -------------------------------------------------------------
def test_croissant_record_is_valid_jsonld():
    rec = feature_table_croissant(
        "VidAudit-GenVidBench",
        description="Per-clip features + LOGO/RvR splits for the audited benchmark.",
        feature_files=[{"id": "temporalspec.csv", "name": "temporalspec.csv",
                        "contentUrl": "https://example.org/temporalspec.csv", "sha256": "ab" * 32}],
        version="0.1.0", license="https://creativecommons.org/licenses/by/4.0/",
        recipe={"canonical": recipe_id(), "kfilter": {"min_frames": 16}},
    )
    assert rec["@type"] == "sc:Dataset"
    assert rec["conformsTo"].endswith("croissant/1.0")
    assert rec["distribution"][0]["sha256"] == "ab" * 32
    fields = {f["name"] for f in rec["recordSet"][0]["field"]}
    assert {"video_id", "generator", "is_real"} <= fields
    assert rec["dct:provenance"]["canonical"].startswith("h264-")
    json.dumps(rec)                                                    # must be serializable


# ---- fetch guards ----------------------------------------------------------
def test_download_file_rejects_non_http(tmp_path):
    with pytest.raises(RuntimeError):
        download_file("/some/local/path", str(tmp_path / "x"))


def test_byo_instructions_mentions_redistribution():
    assert "redistributable" in byo_instructions("GenVidBench")


def test_download_file_cache_hit_skips_network(tmp_path):
    """A cached file whose sha256 matches is returned without touching the network
    (the url here is bogus; if it were fetched the test would error)."""
    import hashlib
    dst = tmp_path / "cached.bin"
    dst.write_bytes(b"vidaudit")
    sha = hashlib.sha256(b"vidaudit").hexdigest()
    assert download_file("https://example.invalid/never", str(dst), sha256=sha) == str(dst)


# ---- end-to-end reconstruct (needs ffmpeg) ---------------------------------
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_reconstruct_canonicalizes_and_filters(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for vid, dur in [("a", 2.5), ("b", 3.0)]:
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"testsrc=size=64x64:rate=8:duration={dur}",
                        "-pix_fmt", "yuv420p", str(src / f"{vid}.mp4")],
                       check=True, capture_output=True)
    man = tmp_path / "prov.csv"
    pd.DataFrame([
        {"video_id": "a", "generator": "real_x", "label": "real", "is_real": 1, "src_path": "a.mp4"},
        {"video_id": "b", "generator": "g1", "label": "gen", "is_real": 0, "src_path": "b.mp4"},
    ]).to_csv(man, index=False)

    out = reconstruct(str(man), str(tmp_path / "out"), sources_dir=str(src), kfilter=KFilter(min_frames=8))
    df = pd.read_csv(out)
    assert list(df.columns) == ["video_id", "generator", "label", "is_real", "mp4_path"]
    assert len(df) == 2 and all(os.path.exists(p) for p in df["mp4_path"])
    assert frame_count(df["mp4_path"].iloc[0]) >= 8
    prov = json.load(open(tmp_path / "out" / "provenance.json"))
    assert prov["recipe"].startswith("h264-") and prov["kfilter"]["n_kept"] == 2
