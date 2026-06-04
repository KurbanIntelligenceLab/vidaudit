"""Reconstruct a dataset locally - the released package is features + recipe + a
provenance manifest, never the source videos (copyright). A user brings the source
clips (downloaded from the original benchmark) and `reconstruct` applies the
canonical re-encode (P1) and the length filter (P2) to produce a ready-to-extract
manifest plus a provenance record, so everyone evaluates on byte-identical clips.

`download_file` (verify + cache) covers the rare dataset whose clips *are*
redistributable. Re-encoding a real dataset is a cluster job.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

import pandas as pd

from vidaudit.data.canonical import CANONICAL, CanonicalSpec, canonicalize, recipe_id
from vidaudit.data.filters import KFilter

READY_COLS = ["video_id", "generator", "label", "is_real", "mp4_path"]


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download_file(url: str, dst: str, sha256: Optional[str] = None) -> str:
    """Download `url` -> `dst` (skip if cached + verified), then sha256-verify."""
    if not url.lower().startswith(("http://", "https://")):
        raise RuntimeError(f"not an http(s) url: {url!r}. For host-restricted sources, "
                           f"download manually and use `reconstruct(..., sources_dir=...)`.")
    if os.path.exists(dst) and (sha256 is None or sha256_of(dst) == sha256):
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    import urllib.request
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    if sha256 and sha256_of(tmp) != sha256:
        os.remove(tmp)
        raise RuntimeError(f"sha256 mismatch for {url!r}")
    os.replace(tmp, dst)
    return dst


def reconstruct(manifest_in: str, out_root: str, *, sources_dir: Optional[str] = None,
                spec: CanonicalSpec = CANONICAL, kfilter: Optional[KFilter] = None,
                src_col: str = "src_path") -> str:
    """Rebuild a ready-to-extract manifest from a provenance manifest + local sources.

    `manifest_in` columns: video_id, generator, label, is_real, and either `src_col`
    (path relative to `sources_dir`, or absolute) or `url` (redistributable sources).
    Writes canonical clips to `out_root/canonical/`, the ready manifest to
    `out_root/manifest.csv`, and `out_root/provenance.json`. Returns the manifest path.
    """
    df = pd.read_csv(manifest_in)
    os.makedirs(out_root, exist_ok=True)
    canon_dir = os.path.join(out_root, "canonical")

    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        if d.get("url"):
            src = download_file(d["url"], os.path.join(out_root, "_src", f"{d['video_id']}.mp4"))
        else:
            rel = d[src_col]
            src = rel if os.path.isabs(str(rel)) else os.path.join(sources_dir or ".", str(rel))
        if not os.path.exists(src):
            raise FileNotFoundError(f"source missing for {d['video_id']!r}: {src}. "
                                    f"{byo_instructions()}")
        dst = canonicalize(src, os.path.join(canon_dir, f"{d['video_id']}.mp4"), spec)
        rows.append({"video_id": d["video_id"], "generator": d["generator"],
                     "label": d.get("label", ""), "is_real": int(d["is_real"]), "mp4_path": dst})

    ready = pd.DataFrame(rows, columns=READY_COLS)
    report: Dict = {"recipe": recipe_id(spec), "n_clips": int(len(ready))}
    if kfilter is not None:
        ready, kreport = kfilter.apply(ready)
        report["kfilter"] = kreport

    man_path = os.path.join(out_root, "manifest.csv")
    ready.to_csv(man_path, index=False)
    with open(os.path.join(out_root, "provenance.json"), "w") as f:
        json.dump(report, f, indent=2)
    return man_path


def byo_instructions(name: str = "the dataset") -> str:
    return (f"Source videos for {name} are not redistributable. Download them from the "
            f"original benchmark, then point `reconstruct(manifest_in, out_root, "
            f"sources_dir=<your download dir>)` at them; the provenance manifest's "
            f"`src_path` column gives each clip's relative path. P1 (canonical re-encode) "
            f"+ P2 (length filter) then produce byte-identical clips for everyone.")
