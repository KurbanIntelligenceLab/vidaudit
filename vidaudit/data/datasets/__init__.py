"""Dataset plugin API + registry.

Adding a benchmark is a thin adapter: subclass `VideoDataset`, set a `DatasetSpec`,
implement `clips()`, and decorate with `@register_dataset`. The audit pipeline is
dataset-agnostic, so it is then available everywhere (`--dataset <name>`, and
combinable as `--dataset a+b`).
"""
from .base import (  # noqa: F401
    DatasetSpec,
    VideoDataset,
    all_datasets,
    get_dataset,
    register_dataset,
)
