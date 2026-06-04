"""Standardized training subsystem.

`TrainConfig` + `apply_overrides` are torch-free (a detector's
`default_train_config()` returns a `TrainConfig` without importing torch). The
loop (`SupervisedTrainer`) and the component registries live in submodules and
import torch lazily, so importing this package — or the detector contract — does
not require torch.
"""
from __future__ import annotations

from vidaudit.train.config import TrainConfig, apply_overrides

__all__ = ["TrainConfig", "apply_overrides"]
