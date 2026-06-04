"""Training subsystem: config overrides, the name-addressable registries, and a
tiny end-to-end train -> checkpoint -> reload -> predict smoke on synthetic
features (a sub-second CPU MLP; heavy training runs on the cluster)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from vidaudit.train.config import TrainConfig, apply_overrides


def test_apply_overrides_coerces_by_field_type():
    cfg = TrainConfig()
    apply_overrides(cfg, ["lr=3e-4", "epochs=100", "hidden=512,256,128",
                          "amp=false", "loss=focal", "focal_gamma=1.5"])
    assert cfg.lr == pytest.approx(3e-4) and isinstance(cfg.lr, float)
    assert cfg.epochs == 100 and isinstance(cfg.epochs, int)
    assert cfg.hidden == (512, 256, 128)
    assert cfg.amp is False
    assert cfg.loss == "focal"
    assert cfg.extra["focal_gamma"] == 1.5          # unknown key -> extra, coerced to float


def test_apply_overrides_rejects_malformed_pair():
    with pytest.raises(ValueError):
        apply_overrides(TrainConfig(), ["lr"])


def test_registries_build_components():
    import torch

    from vidaudit.train import registries as reg
    cfg = TrainConfig(head="mlp", hidden=(8,), dropout=0.0, lr=1e-3)
    head = reg.build_head("mlp", in_dim=13, cfg=cfg)
    out = head(torch.zeros(4, 13))
    assert out.shape == (4,)                          # one logit per clip, squeezed
    assert reg.build_head("linear", 13, cfg)(torch.zeros(2, 13)).shape == (2,)

    loss = reg.build_loss("bce", cfg)
    assert float(loss(torch.zeros(3), torch.ones(3))) > 0
    assert float(reg.build_loss("focal", cfg)(torch.zeros(3), torch.ones(3))) > 0

    opt = reg.build_optimizer("adamw", head.parameters(), cfg)
    assert reg.build_scheduler("cosine", opt, cfg) is not None
    assert reg.build_scheduler("none", opt, cfg) is None
    with pytest.raises(KeyError):
        reg.build_loss("nope", cfg)


def _synth_features(path: str, n_per: int = 80, d: int = 13, shift: float = 1.6) -> pd.DataFrame:
    """Real ~ N(0,1), generated ~ N(shift,1): a clean signal so a probe should learn it."""
    rng = np.random.default_rng(0)
    rows = []
    for src in ("real_a", "real_b"):
        X = rng.standard_normal((n_per, d))
        for i, x in enumerate(X):
            rows.append({"video_id": f"{src}_{i}", "generator": src, "is_real": 1,
                         **{f"f_{j}": v for j, v in enumerate(x)}})
    for gen in ("g1", "g2", "g3"):
        X = rng.standard_normal((n_per, d)) + shift
        for i, x in enumerate(X):
            rows.append({"video_id": f"{gen}_{i}", "generator": gen, "is_real": 0,
                         **{f"f_{j}": v for j, v in enumerate(x)}})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def test_train_smoke_end_to_end(tmp_path):
    from vidaudit.detectors.registry import get
    csv = str(tmp_path / "feats.csv")
    df = _synth_features(csv)

    det = get("mlp-probe")
    assert det.is_trainable and det.spec.trainable
    cfg = det.default_train_config()
    cfg.features = csv
    cfg.epochs, cfg.batch_size = 30, 64
    out_dir = str(tmp_path / "run")
    ckpt = det.train(cfg, [], out_dir)

    assert os.path.exists(ckpt)
    metrics = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert metrics["best_val_auc"] > 0.8             # the signal is learnable
    assert len(metrics["history"]) == 30

    det.load_weights(ckpt)                            # reload the trained head...
    scores = det.predict_table(df)                    # ...and score the table
    assert scores.shape == (len(df),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # generated rows should score higher (more generated) than real rows
    y = (1 - df["is_real"]).to_numpy()
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_train_alternative_components(tmp_path):
    """Exercise the non-default registry entries end-to-end: linear head + focal loss
    + SGD + step scheduler (the smoke covers mlp/bce/adamw/cosine)."""
    from vidaudit.detectors.registry import get
    csv = str(tmp_path / "feats.csv")
    _synth_features(csv)
    det = get("mlp-probe")
    cfg = det.default_train_config()
    cfg.features = csv
    cfg.head, cfg.loss, cfg.optimizer, cfg.scheduler = "linear", "focal", "sgd", "step"
    cfg.lr, cfg.epochs, cfg.batch_size = 0.05, 40, 64
    out_dir = str(tmp_path / "run")
    det.train(cfg, [], out_dir)
    m = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert len(m["history"]) == 40 and m["best_val_auc"] > 0.65


def test_restrav_trainable_head(tmp_path):
    """A published backbone detector (ReStraV) trains a head over its 21-d features
    through the same trainer (build_model + default_train_config), no DINOv2 needed."""
    from vidaudit.detectors.registry import get
    det = get("restrav")
    assert det.is_trainable and det.spec.trainable
    assert det.default_train_config().hidden == (64,)        # ReStraV-specific recipe
    csv = str(tmp_path / "restrav_feats.csv")
    _synth_features(csv, d=21)                                # 21-d, like ReStraV features()
    cfg = det.default_train_config()
    cfg.features, cfg.epochs, cfg.batch_size = csv, 30, 64
    out_dir = str(tmp_path / "run")
    det.train(cfg, [], out_dir)
    m = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert m["best_val_auc"] > 0.7


def test_train_with_separate_val_features(tmp_path):
    """cfg.val_features uses a held-out table instead of splitting the train table."""
    from vidaudit.detectors.registry import get
    tr, va = str(tmp_path / "tr.csv"), str(tmp_path / "va.csv")
    _synth_features(tr)
    _synth_features(va, n_per=30)
    det = get("mlp-probe")
    cfg = det.default_train_config()
    cfg.features, cfg.val_features = tr, va
    cfg.epochs, cfg.batch_size = 15, 64
    out_dir = str(tmp_path / "run")
    det.train(cfg, [], out_dir)
    m = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert len(m["history"]) == 15 and m["best_val_auc"] > 0.6
