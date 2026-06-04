"""Name-addressable registries for the standard trainer.

Loss / optimizer / scheduler / head are looked up by name so a recipe (or a
`--set loss=focal` override) swaps a component without touching code. Add a new
component by decorating a builder; it is immediately addressable from any recipe.

Builders take the minimal context they need (a `TrainConfig`, parameters, an
optimizer) and return the constructed object. torch is imported here because the
training path requires it; the detector contract and the audit stay torch-free.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

# ---- registries ------------------------------------------------------------
LOSSES: Dict[str, Callable] = {}
OPTIMIZERS: Dict[str, Callable] = {}
SCHEDULERS: Dict[str, Callable] = {}
HEADS: Dict[str, Callable] = {}


def _register(table: Dict[str, Callable], name: str):
    def deco(fn: Callable) -> Callable:
        table[name] = fn
        return fn
    return deco


def register_loss(name): return _register(LOSSES, name)
def register_optimizer(name): return _register(OPTIMIZERS, name)
def register_scheduler(name): return _register(SCHEDULERS, name)
def register_head(name): return _register(HEADS, name)


def _get(table: Dict[str, Callable], name: str, kind: str) -> Callable:
    try:
        return table[name]
    except KeyError:
        raise KeyError(f"unknown {kind} {name!r}; registered: {sorted(table)}") from None


# ---- heads -----------------------------------------------------------------
class MLPHead(nn.Module):
    """A 1-logit classifier head over a fixed-dim feature vector. hidden=() is a
    plain linear probe. Output is a single logit per clip (p(generated) after a
    sigmoid), matching the generated-positive label convention."""

    def __init__(self, in_dim: int, hidden=(256,), dropout: float = 0.1):
        super().__init__()
        layers, d = [], int(in_dim)
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register_head("mlp")
def _head_mlp(in_dim, cfg):
    return MLPHead(in_dim, hidden=cfg.hidden, dropout=cfg.dropout)


@register_head("linear")
def _head_linear(in_dim, cfg):
    return MLPHead(in_dim, hidden=(), dropout=0.0)


def build_head(name: str, in_dim: int, cfg) -> nn.Module:
    return _get(HEADS, name, "head")(in_dim, cfg)


# ---- losses (callable(logits, target_float) -> scalar) ---------------------
@register_loss("bce")
def _loss_bce(cfg):
    return nn.BCEWithLogitsLoss()


@register_loss("focal")
def _loss_focal(cfg):
    gamma = float(cfg.extra.get("focal_gamma", 2.0))
    alpha = float(cfg.extra.get("focal_alpha", 0.25))

    def focal(logits, target):
        # binary focal loss on a single logit; stable form via BCE-with-logits
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        a_t = alpha * target + (1 - alpha) * (1 - target)
        return (a_t * (1 - p_t).pow(gamma) * ce).mean()

    return focal


def build_loss(name: str, cfg) -> Callable:
    return _get(LOSSES, name, "loss")(cfg)


# ---- optimizers ------------------------------------------------------------
@register_optimizer("adamw")
def _opt_adamw(params, cfg):
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


@register_optimizer("adam")
def _opt_adam(params, cfg):
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


@register_optimizer("sgd")
def _opt_sgd(params, cfg):
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, nesterov=cfg.momentum > 0)


def build_optimizer(name: str, params, cfg):
    return _get(OPTIMIZERS, name, "optimizer")(params, cfg)


# ---- schedulers (return a scheduler or None) -------------------------------
@register_scheduler("cosine")
def _sched_cosine(opt, cfg):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))


@register_scheduler("step")
def _sched_step(opt, cfg):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.epochs // 3), gamma=0.1)


@register_scheduler("none")
def _sched_none(opt, cfg):
    return None


def build_scheduler(name: str, opt, cfg):
    return _get(SCHEDULERS, name, "scheduler")(opt, cfg)
