"""Clip decoding (PyAV) + batch extraction into the audit's feature-table schema.

A researcher runs:
    python run.py extract <detector> --manifest clips.csv --out feats.csv
    python run.py eval --features feats.csv

The manifest is a CSV with columns (video_id, generator, label, is_real, mp4_path).
`extract_table` calls the detector's features() (or score()) on each clip and writes
the same schema the audit harness consumes, with resumable checkpointing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from vidaudit.detectors.base import Clip, Detector


def decode_clip(path, n_frames: int = 8, size: int = 224) -> np.ndarray:
    """Decode an mp4 to a uniform [n_frames, size, size, 3] uint8 RGB array.

    Uses PyAV (the env's decoder). Short clips repeat the last frame; long clips
    sample at uniform stride. (The cluster scripts used decord; PyAV is the unified
    choice here, so re-extracted features can drift negligibly from the original
    CSVs because the decode/resize path differs.)
    """
    import av
    import cv2

    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        raise RuntimeError(f"empty decode: {path}")
    T = len(frames)
    if T <= n_frames:
        idx = list(range(T)) + [T - 1] * (n_frames - T)
    else:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int).tolist()
    out = np.empty((n_frames, size, size, 3), np.uint8)
    for k, j in enumerate(idx):
        fr = frames[j]
        if fr.shape[:2] != (size, size):
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR)
        out[k] = fr
    return out


def clips_from_manifest(csv_path) -> List[Clip]:
    """Build Clip objects from a manifest CSV (video_id, generator, label, is_real, mp4_path)."""
    df = pd.read_csv(csv_path)
    path_col = "mp4_path" if "mp4_path" in df.columns else "path"
    clips = []
    for _, r in df.iterrows():
        clips.append(Clip(
            video_id=str(r["video_id"]),
            path=str(r[path_col]),
            source=str(r.get("generator", r.get("source", ""))),
            is_real=int(r.get("is_real", 0)),
            meta={"label": r.get("label", "")},
        ))
    return clips


def extract_table(detector: Detector, clips: Iterable[Clip], *, kind: str = "auto",
                  out: Optional[str] = None, checkpoint_every: int = 200,
                  verbose: bool = True) -> pd.DataFrame:
    """Extract features (or score) for each clip into the audit's table schema.

    kind: "features" | "score" | "auto" (features if available, else score). Feature
    columns are named `<detector>_<j>`; the score path writes a single `score` column.
    Resumable: existing `out` rows are kept and skipped.
    """
    prefix = detector.spec.name.lower().replace("-", "").replace("+", "")
    use_feats = detector.has_features if kind == "auto" else (kind == "features")
    if use_feats and not detector.has_features:
        raise ValueError(f"{detector.spec.name} has no features(); use --kind score")
    if not use_feats and not detector.has_native_head:
        raise ValueError(f"{detector.spec.name} has no score(); use --kind features")

    done: set = set()
    rows: List[dict] = []
    if out and Path(out).exists():
        prev = pd.read_csv(out)
        done = set(prev["video_id"].astype(str))
        rows = prev.to_dict("records")

    n_fail = 0
    for clip in clips:
        if clip.video_id in done:
            continue
        try:
            rec = {"video_id": clip.video_id, "generator": clip.source,
                   "label": clip.meta.get("label", ""), "is_real": int(clip.is_real),
                   "mp4_path": clip.path}
            if use_feats:
                v = np.asarray(detector.features(clip), dtype=float).ravel()
                for j, val in enumerate(v):
                    rec[f"{prefix}_{j}"] = float(val)
            else:
                rec["score"] = float(detector.score(clip))
            rows.append(rec)
        except Exception as e:
            n_fail += 1
            if verbose:
                print(f"[extract] FAIL {clip.video_id}: {type(e).__name__}: {e}", flush=True)
            continue
        if out and checkpoint_every and (len(rows) % checkpoint_every == 0):
            pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    if verbose:
        print(f"[extract] {detector.spec.name}: {len(df)} rows, {n_fail} fails"
              + (f" -> {out}" if out else ""), flush=True)
    return df
