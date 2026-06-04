"""Dataset plugin: a benchmark is a thin adapter behind a registry.

We never redistribute source videos (copyright). Instead, a dataset adapter knows
the **native download layout** of the original release and maps it into the audit's
schema. `scan(root)` walks a user's local download (obtained from `spec.download`)
and yields `Clip` objects with provenance (source, is_real, video_id); the
preprocessing tool (`vidaudit.data.fetch.prepare_dataset`) then applies P1 (canonical
re-encode) and P2 (length filter) to produce a ready-to-extract manifest. The audit
pipeline is dataset-agnostic, so a new benchmark gets the metrics, the verdict, and a
leaderboard section for free. Two or more real sources enable the real-vs-real floor
(P3); fewer auto-disable it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from vidaudit.detectors.base import Clip


@dataclass(slots=True)
class DatasetSpec:
    name: str
    generators: List[str] = field(default_factory=list)
    real_sources: List[str] = field(default_factory=list)
    has_official_split: bool = False
    license_note: str = ""
    homepage: str = ""                 # official project page / paper
    download: str = ""                 # where + how to obtain the source videos (we do not mirror)
    fetch: Optional[Callable] = None   # optional auto-download recipe, when a release permits it

    @property
    def supports_rvr(self) -> bool:
        """P3 (real-vs-real) needs at least two distinct real sources."""
        return len(self.real_sources) >= 2


class VideoDataset(ABC):
    """Subclass + @register_dataset to make a benchmark available everywhere."""

    spec: DatasetSpec

    @abstractmethod
    def scan(self, root: str) -> Iterable[Clip]:
        """Enumerate the dataset from its native download layout at `root`, yielding a
        `Clip` per video with the *original* file path and provenance set
        (`source`, `is_real`, `video_id`, `dataset`). A clip backed by a zip member
        carries `meta={'zip': <zip path>, 'member': <name>}` and `path=<zip path>`.
        We do not redistribute the videos; users download them from `spec.download`."""
        raise NotImplementedError

    def provenance(self) -> dict:
        return {
            "name": self.spec.name,
            "homepage": self.spec.homepage,
            "download": self.spec.download,
            "generators": list(self.spec.generators),
            "real_sources": list(self.spec.real_sources),
            "supports_rvr": self.spec.supports_rvr,
        }


# --- registry ---
_REGISTRY: dict = {}


def register_dataset(name: str):
    """Class decorator: register a VideoDataset subclass under `name`."""
    def _deco(cls):
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise KeyError(f"dataset name already registered: {name!r}")
        _REGISTRY[key] = cls
        return cls
    return _deco


def get_dataset(name: str) -> VideoDataset:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def all_datasets() -> List[str]:
    return sorted(_REGISTRY)
