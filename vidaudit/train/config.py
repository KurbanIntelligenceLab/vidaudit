"""Training configuration + override parsing.

`TrainConfig` is the single, fully-overridable hyperparameter object the standard
trainer consumes. Defaults for a method live in `scripts/train/<name>.sh` (which
just calls `run.py train <name> --set k=v ...`); a detector's
`default_train_config()` returns the in-code defaults. Every knob is addressable
by name through `--set`, and loss / optimizer / scheduler / head are *names* that
resolve against the registries in `vidaudit.train.registries`.

This module is deliberately torch-free so the contract (and `default_train_config`)
loads without importing torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Sequence, Tuple


@dataclass(slots=True)
class TrainConfig:
    # --- data (a precomputed per-clip feature table; the `extract` output) -----
    features: str = ""                       # path to the training feature CSV
    val_features: str = ""                   # optional held-out CSV; else split `features`
    feature_cols: Optional[str] = None       # None=auto; "auto:prefix=v,a"; or explicit list
    subset: Optional[str] = None             # matched-cell CSV (video_id, generator)
    val_frac: float = 0.2                    # in-table validation fraction (checkpoint selection)
    arrow: bool = False                      # read the table with the pyarrow backend

    # --- model head (resolved via registries.build_head) ----------------------
    head: str = "mlp"                        # "linear" | "mlp"
    hidden: Tuple[int, ...] = (256,)         # MLP hidden widths
    dropout: float = 0.1

    # --- optimization (all resolve against the registries) --------------------
    loss: str = "bce"                        # "bce" | "focal"
    optimizer: str = "adamw"                 # "adamw" | "adam" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9                    # sgd only
    scheduler: str = "cosine"                # "cosine" | "step" | "none"
    epochs: int = 50
    batch_size: int = 256
    amp: bool = True                         # mixed precision (cuda only; ignored elsewhere)
    grad_clip: float = 1.0                   # max grad norm; 0 disables

    # --- bookkeeping ----------------------------------------------------------
    seed: int = 42
    device: str = "auto"                     # "auto" | "cuda" | "mps" | "cpu"
    num_workers: int = 0

    # --- escape hatch: method-specific knobs forwarded to build_model ---------
    extra: dict = field(default_factory=dict)


def _coerce(current, value: str):
    """Coerce a CLI string to the type of the field's current value."""
    if isinstance(current, bool):                       # bool before int (bool is an int)
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, tuple):
        return tuple(int(x) for x in value.split(",") if x.strip() != "")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value                                        # str / None stay strings


def _coerce_scalar(value: str):
    """Best-effort coercion for unknown (extra.*) keys: int -> float -> bool -> str."""
    v = value.strip()
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return value


def apply_overrides(cfg: TrainConfig, pairs: Sequence[str]) -> TrainConfig:
    """Apply `key=value` overrides in place. Known fields are type-coerced to the
    field's type; unknown keys land in `cfg.extra` (best-effort coerced). Raises on
    a malformed pair so a typo in a recipe fails loud instead of silently no-op'ing."""
    names = {f.name for f in fields(cfg)} - {"extra"}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key in names:
            setattr(cfg, key, _coerce(getattr(cfg, key), value))
        else:
            cfg.extra[key] = _coerce_scalar(value)
    return cfg
