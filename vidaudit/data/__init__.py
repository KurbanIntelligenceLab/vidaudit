"""Standardized data package.

A benchmark is a thin adapter (`datasets/`); the audit is dataset-agnostic. What
VidAudit redistributes is per-clip features + splits + provenance + a reproducible
recipe, never the source videos. The recipe is two controls:

  * P1 canonical re-encode (`canonical.py`) - normalize codec/container fingerprints
  * P2 clip-length filter (`filters.py`)    - measure + remove duration leakage

`fetch.reconstruct` applies both to a user's local source clips to yield a
ready-to-extract manifest; `croissant.py` emits machine-readable metadata for the
released feature tables.
"""
from vidaudit.data.canonical import (  # noqa: F401
    CANONICAL, CanonicalSpec, canonicalize, ffmpeg_cmd, recipe_id,
)
from vidaudit.data.croissant import (  # noqa: F401
    feature_table_croissant, write_croissant,
)
from vidaudit.data.datasets import (  # noqa: F401
    DatasetSpec, VideoDataset, all_datasets, get_dataset, register_dataset,
)
from vidaudit.data.fetch import (  # noqa: F401
    byo_instructions, download_file, reconstruct,
)
from vidaudit.data.filters import KFilter, frame_count  # noqa: F401
