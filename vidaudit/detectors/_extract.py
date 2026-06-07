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

# Cap how many frames we materialize per clip before sampling. Detectors only need a
# handful of uniformly sampled frames, but decoding an entire long/high-res clip into RAM
# spikes memory (a single 4K or multi-minute clip can reach many GB and OOM a batch run).
# Clips at or under this length are unaffected; longer clips are sampled from this window.
_MAX_DECODE = 512


def _decode_capped(container):
    """Decode RGB frames up to _MAX_DECODE (bounds per-clip memory)."""
    out = []
    for f in container.decode(video=0):
        out.append(f.to_ndarray(format="rgb24"))
        if len(out) >= _MAX_DECODE:
            break
    return out


def decode_clip(path, n_frames: int = 8, size: int = 224, window_sec=None) -> np.ndarray:
    """Decode an mp4 to a [n_frames, size, size, 3] uint8 RGB array.

    Uses PyAV (the env's decoder). With `window_sec` set, samples uniformly across a
    `window_sec`-second window centred on the clip midpoint (ReStraV protocol);
    otherwise samples uniformly across the whole clip. Short clips repeat the last
    frame. (The cluster scripts used decord; PyAV is the unified choice here, so
    re-extracted features can drift negligibly from the original CSVs because the
    decode/resize path differs.)
    """
    import av
    import cv2

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        frames = _decode_capped(container)
    if not frames:
        raise RuntimeError(f"empty decode: {path}")
    T = len(frames)
    if window_sec is not None and T > n_frames and fps > 0:
        dur = T / fps
        center = dur / 2.0
        half = min(window_sec / 2.0, center, dur - center)
        if half <= 0:
            idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        else:
            sf = max(0, int(round((center - half) * fps)))
            ef = min(T - 1, int(round((center + half) * fps)))
            idx = (np.full(n_frames, sf, int) if ef <= sf
                   else np.linspace(sf, ef, n_frames).round().astype(int))
        idx = np.clip(idx, 0, T - 1).tolist()
    elif T <= n_frames:
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


def decode_pil(path, n_frames: int = 12, window_sec=None):
    """Decode n_frames as native-resolution PIL RGB images.

    For methods that do their own resize/crop (CLIP's processor, WaveRep's center-crop),
    so we must not pre-distort to a square. With `window_sec` set, samples across a
    centred window (WaveRep/ReStraV protocol); otherwise across the whole clip.
    """
    import av
    from PIL import Image
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        frames = _decode_capped(container)
    if not frames:
        raise RuntimeError(f"empty decode: {path}")
    T = len(frames)
    if window_sec is not None and T > n_frames and fps > 0:
        dur = T / fps
        center = dur / 2.0
        half = min(window_sec / 2.0, center, dur - center)
        if half <= 0:
            idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        else:
            sf = max(0, int(round((center - half) * fps)))
            ef = min(T - 1, int(round((center + half) * fps)))
            idx = (np.full(n_frames, sf, int) if ef <= sf
                   else np.linspace(sf, ef, n_frames).round().astype(int))
        idx = np.clip(idx, 0, T - 1).tolist()
    elif T >= n_frames:
        idx = np.linspace(0, T - 1, n_frames).round().astype(int).tolist()
    else:
        idx = list(range(T)) + [T - 1] * (n_frames - T)
    return [Image.fromarray(frames[int(j)]) for j in idx]


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
        try:
            prev = pd.read_csv(out)
            done = set(prev["video_id"].astype(str))
            rows = prev.to_dict("records")
        except Exception:
            done, rows = set(), []  # empty/corrupt prior file: start fresh

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
                names = getattr(detector, "feature_names", None)
                cols = (list(names) if names and len(names) == len(v)
                        else [f"{prefix}_{j}" for j in range(len(v))])
                for c, val in zip(cols, v):
                    rec[c] = float(val)
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
