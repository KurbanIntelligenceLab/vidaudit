"""AIGVDBench (Ma et al., 2026; "Your One-Stop Solution for AI-Generated Video
Detection", arXiv:2601.11035) adapter.

Native download layout (a `huggingface_hub` snapshot of the AIGVDBench/AIGVDBench
dataset repo; videos are NOT redistributed by us):

    <root>/AIGVDBench/Real/Real.zip                         -> real videos
    <root>/AIGVDBench/OpenSource/<category>/<Model>.zip      -> generated (per model)
    <root>/Split/{train,val,test}.jsonl                     -> prompt index (Video_id, prompt)

The label and the generator come from the PATH (Real vs OpenSource/<cat>/<Model>), not
the jsonl. The videos live inside per-model zips, so we enumerate mp4 members directly
(stdlib zipfile), no manual unzip needed, and also tolerate an already-extracted tree.
One real source ("Real"), so the real-vs-real floor (P3) auto-disables.
"""
from __future__ import annotations

import os
import zipfile
from typing import Iterable, Optional

from vidaudit.data.datasets.base import DatasetSpec, VideoDataset, register_dataset
from vidaudit.detectors.base import Clip


def _data_root(root: str) -> Optional[str]:
    """Find the dir holding Real/ + OpenSource/ (accepts the snapshot dir, the
    AIGVDBench/ dir, or a HuggingFace cache root)."""
    cands = [root, os.path.join(root, "AIGVDBench")]
    cache = os.path.join(root, "snapshots")
    if os.path.isdir(cache):
        for h in sorted(os.listdir(cache)):
            cands.append(os.path.join(cache, h, "AIGVDBench"))
    for c in cands:
        if os.path.isdir(os.path.join(c, "Real")) or os.path.isdir(os.path.join(c, "OpenSource")):
            return c
    return None


def _mp4_members(zip_path: str):
    try:
        with zipfile.ZipFile(zip_path) as z:
            return [n for n in z.namelist() if n.lower().endswith(".mp4") and not n.endswith("/")]
    except zipfile.BadZipFile:
        return []


@register_dataset("aigvdbench")
class AIGVDBench(VideoDataset):
    spec = DatasetSpec(
        name="aigvdbench",
        generators=[],                      # 31 models; discovered from OpenSource/<cat>/<Model>
        real_sources=["Real"],              # one real source -> RvR (P3) auto-disables
        has_official_split=True,
        license_note="CC-BY-4.0 (AIGVDBench release); source videos not redistributable.",
        homepage="https://github.com/LongMa-2025/AIGVDBench",
        download=("Download the AIGVDBench/AIGVDBench dataset from HuggingFace "
                  "(huggingface_hub snapshot_download, repo_type='dataset'), then point "
                  "`prepare-data --dataset aigvdbench --source <snapshot dir>` at it. "
                  "Videos are read straight from the per-model zips; no manual unzip needed."),
    )

    def scan(self, root: str) -> Iterable[Clip]:
        base = _data_root(root)
        if base is None:
            raise FileNotFoundError(
                f"no AIGVDBench layout under {root!r}. Expected AIGVDBench/Real/ + "
                f"AIGVDBench/OpenSource/<category>/<Model>.zip (a HuggingFace snapshot). "
                f"See `spec.download`.")

        # real: Real/*.zip (or an extracted Real/ tree)
        yield from self._scan_group(os.path.join(base, "Real"), source="Real", is_real=1, category="real")
        # generated: OpenSource/<category>/<Model>.zip (T2V, I2V, V2V, ...)
        osrc = os.path.join(base, "OpenSource")
        if os.path.isdir(osrc):
            for cat in sorted(os.listdir(osrc)):
                cdir = os.path.join(osrc, cat)
                if os.path.isdir(cdir):
                    yield from self._scan_group(cdir, source=None, is_real=0, category=cat)

    def _scan_group(self, group_dir: str, *, source, is_real: int, category: str) -> Iterable[Clip]:
        if not os.path.isdir(group_dir):
            return
        for entry in sorted(os.listdir(group_dir)):
            path = os.path.join(group_dir, entry)
            if entry.lower().endswith(".zip"):
                model = source or os.path.splitext(entry)[0]      # <Model>.zip -> model name
                for member in _mp4_members(path):
                    yield Clip(video_id=os.path.splitext(os.path.basename(member))[0],
                               path=path, source=model, is_real=is_real, dataset="aigvdbench",
                               meta={"zip": path, "member": member, "category": category})
            elif os.path.isdir(path):                              # already-extracted <Model>/ tree
                model = source or entry
                for r, _d, files in os.walk(path):
                    for fn in sorted(files):
                        if fn.lower().endswith(".mp4"):
                            yield Clip(video_id=os.path.splitext(fn)[0],
                                       path=os.path.join(r, fn), source=model, is_real=is_real,
                                       dataset="aigvdbench", meta={"category": category})
