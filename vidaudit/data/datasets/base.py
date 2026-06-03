"""Dataset plugin: a benchmark is a thin adapter behind a registry.

Implement `clips()` to yield `Clip` objects with provenance, and declare the
generators and real sources. The audit pipeline (P1-P6, matched cells, LOGO) is
dataset-agnostic, so a new dataset gets the metrics, the verdict, and a
leaderboard section for free. Two real sources or more enable the real-vs-real
floor (P3); fewer auto-disable it (e.g. AIGVDBench ships a single real source).
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
    fetch: Optional[Callable] = None   # recipe to reconstruct clips locally (no redistribution)

    @property
    def supports_rvr(self) -> bool:
        """P3 (real-vs-real) needs at least two distinct real sources."""
        return len(self.real_sources) >= 2


class VideoDataset(ABC):
    """Subclass + @register_dataset to make a benchmark available everywhere."""

    spec: DatasetSpec

    @abstractmethod
    def clips(self, split: str = "logo", cell: str = "matched") -> Iterable[Clip]:
        """Yield Clip objects for the requested split/cell (with provenance set)."""
        raise NotImplementedError

    def provenance(self) -> dict:
        return {
            "name": self.spec.name,
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
