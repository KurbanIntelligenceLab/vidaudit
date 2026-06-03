"""Detector registry: `@register("name")` makes a detector discoverable by the
CLI (`run.py eval --model name`) and the leaderboard. Importing a detector module
registers it as a side effect; `aigvaudit.detectors` imports all wrappers.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import Detector

_REGISTRY: Dict[str, Type[Detector]] = {}


def register(name: str):
    """Class decorator: register a Detector subclass under `name`."""
    def _deco(cls: Type[Detector]) -> Type[Detector]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise KeyError(f"detector name already registered: {name!r}")
        _REGISTRY[key] = cls
        return cls
    return _deco


def get(name: str) -> Detector:
    """Instantiate a registered detector by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def all_detectors() -> List[str]:
    return sorted(_REGISTRY)
