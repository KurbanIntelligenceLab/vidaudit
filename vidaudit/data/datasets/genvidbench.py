"""GenVidBench (Ni et al., AAAI 2026; arXiv:2501.11340) adapter.

Native download layout (from https://github.com/genvidbench/GenVidBench, released on
HuggingFace; videos are NOT redistributed by us):

    <root>/Pair1/<source>/<source>/<video_id>.mp4
    <root>/Pair2/<source>/<source>/<video_id>.mp4

`<source>` is a generator (cogvideo, mora, musev, svd, ms, pika, t2vz, vc2) or one of
the two real datasets (vript in Pair1, hd_vg_130m in Pair2). The release ships each
source as a .rar/.7z archive alongside the extracted folder; extract those first, then
point `scan()` at <root>. We tolerate single- or double-nesting under <source>.
"""
from __future__ import annotations

import os
from typing import Iterable

from vidaudit.data.datasets.base import DatasetSpec, VideoDataset, register_dataset
from vidaudit.detectors.base import Clip

_REAL = {"vript", "hd_vg_130m"}
_GENERATORS = ["cogvideo", "mora", "musev", "svd", "ms", "pika", "t2vz", "vc2"]


@register_dataset("genvidbench")
class GenVidBench(VideoDataset):
    spec = DatasetSpec(
        name="genvidbench",
        generators=_GENERATORS,
        real_sources=sorted(_REAL),
        has_official_split=True,
        license_note="GenVidBench release (AAAI 2026); source videos not redistributable.",
        homepage="https://genvidbench.github.io/",
        download=("Download from https://github.com/genvidbench/GenVidBench (HuggingFace "
                  "release / Baidu backup). Extract the per-source .rar/.7z archives so the "
                  "layout is <root>/Pair{1,2}/<source>/<source>/*.mp4, then point "
                  "`prepare-data --dataset genvidbench --source <root>` at it."),
    )

    def __init__(self, pairs=("Pair1", "Pair2")):
        self.pairs = pairs

    def scan(self, root: str) -> Iterable[Clip]:
        seen = False
        for pair in self.pairs:
            pdir = os.path.join(root, pair)
            if not os.path.isdir(pdir):
                continue
            for src in sorted(os.listdir(pdir)):
                sdir = os.path.join(pdir, src)
                if not os.path.isdir(sdir):           # skip the .rar/.7z archive entries
                    continue
                inner = os.path.join(sdir, src)        # GenVidBench double-nests: Pair2/svd/svd/*.mp4
                walk = inner if os.path.isdir(inner) else sdir
                for fn in sorted(os.listdir(walk)):
                    if not fn.lower().endswith(".mp4"):
                        continue
                    seen = True
                    yield Clip(
                        video_id=os.path.splitext(fn)[0],
                        path=os.path.join(walk, fn),
                        source=src,
                        is_real=int(src in _REAL),
                        dataset="genvidbench",
                        meta={"pair": pair},
                    )
        if not seen:
            raise FileNotFoundError(
                f"no GenVidBench clips under {root!r}. Expected "
                f"<root>/Pair{{1,2}}/<source>/<source>/*.mp4 (extract the .rar/.7z archives first). "
                f"See `spec.download`.")
