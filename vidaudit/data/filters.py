"""P2 - clip-length leakage filter.

AIGV benchmarks often differ systematically in clip length by generator (one model
emits 2 s clips, another 4 s, the real set something else). Frame count alone then
separates the classes, so a detector can score well by reading duration instead of
content. P2 measures that leakage and removes it: enforce a common minimum frame
budget so every clip can yield the same K uniformly-sampled frames, and report the
length->label separability before and after, so the control is auditable.

`frame_count` reads container metadata first (cheap) and falls back to decoding.
Counting a full dataset is a cluster job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


def frame_count(path: str) -> int:
    """Number of video frames; container metadata when present, else a decode count."""
    import av
    with av.open(path) as c:
        if not c.streams.video:
            return 0
        st = c.streams.video[0]
        if st.frames:                       # most mp4/h264 expose this without decoding
            return int(st.frames)
        return sum(1 for _ in c.decode(video=0))


def frame_counts(df: pd.DataFrame, path_col: str = "mp4_path") -> pd.Series:
    return df[path_col].map(frame_count)


def _separability(counts, y) -> float:
    """How well clip length alone predicts the generated label, as AUC folded to
    [0.5, 1] (direction-agnostic): 0.5 = no leakage, 1.0 = length fully determines it."""
    from vidaudit.audit.metrics import auc
    import numpy as np
    counts = np.asarray(counts, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = auc(y, counts)
    return float(max(a, 1.0 - a))


@dataclass(slots=True)
class KFilter:
    """Keep clips with at least `min_frames` frames (so K-frame sampling is well-defined)."""
    min_frames: int = 16

    def apply(self, df: pd.DataFrame, path_col: str = "mp4_path",
              counts: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict]:
        df = df.reset_index(drop=True)
        if counts is None:
            counts = frame_counts(df, path_col)
        counts = pd.Series(counts).reset_index(drop=True)
        keep = counts >= self.min_frames
        kept = df[keep].reset_index(drop=True)

        y = (1 - df["is_real"].astype(int)) if "is_real" in df else None
        per_gen = []
        if "generator" in df:
            for g, idx in df.groupby("generator").groups.items():
                cg = counts.loc[list(idx)]
                per_gen.append({"generator": str(g), "n": int(len(cg)),
                                "kept": int(keep.loc[list(idx)].sum()),
                                "median_frames": float(cg.median()),
                                "min_frames": int(cg.min()), "max_frames": int(cg.max())})
        report = {
            "min_frames": self.min_frames,
            "n_in": int(len(df)), "n_kept": int(len(kept)), "n_dropped": int((~keep).sum()),
            "length_leakage_before": _separability(counts, y) if y is not None else None,
            "length_leakage_after": (_separability(counts[keep], y[keep]) if y is not None else None),
            "per_generator": sorted(per_gen, key=lambda r: r["generator"]),
        }
        return kept, report
