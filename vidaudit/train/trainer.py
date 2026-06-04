"""The one supervised loop every trainable detector shares.

`SupervisedTrainer(detector, cfg).fit(train_clips, out_dir)` owns seeding, the
optimizer / loss / scheduler (resolved by name from the registries), AMP, grad
clipping, per-epoch validation AUC, best-checkpoint selection, and writing a
self-describing checkpoint + a metrics history. A detector only supplies
`build_model(cfg)` (the head) and `default_train_config()`; it never reimplements
the loop. Override `Detector.train()` only for a genuinely non-standard recipe.

Heavy training belongs on the cluster (`scripts/train/<name>.sh` -> sbatch); this
loop runs unchanged there. The label is generated-positive (y = 1 - is_real).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
import torch

from vidaudit.audit import metrics as M
from vidaudit.train import data as _data
from vidaudit.train import registries as _reg


def _resolve_device(want: str) -> torch.device:
    if want and want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SupervisedTrainer:
    def __init__(self, detector, cfg):
        self.det = detector
        self.cfg = cfg

    def fit(self, train_clips: Optional[Sequence] = None, out_dir: str = "runs/train") -> str:
        cfg = self.cfg
        if not cfg.features:
            raise ValueError(
                "the standard trainer learns a head over a precomputed feature table; "
                "set cfg.features to an `extract` CSV (or pass --features). Training "
                "directly from raw clips: extract first, then train.")
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = _resolve_device(cfg.device)
        loaders = _data.make_loaders(cfg)
        cfg.extra["in_dim"] = loaders.in_dim                  # hand the head its input width

        model = self.det.build_model(cfg).to(device)
        opt = _reg.build_optimizer(cfg.optimizer, model.parameters(), cfg)
        loss_fn = _reg.build_loss(cfg.loss, cfg)
        sched = _reg.build_scheduler(cfg.scheduler, opt, cfg)
        use_amp = bool(cfg.amp) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc, best_epoch, history = -1.0, -1, []
        ckpt_path = os.path.join(out_dir, "model.pt")
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            for xb, yb in loaders.train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                running += loss.detach().item() * len(xb)
            if sched is not None:
                sched.step()

            val_auc = self._val_auc(model, loaders.val, device)
            history.append({"epoch": epoch, "train_loss": running / max(1, len(loaders.train.dataset)),
                            "val_auc": val_auc, "lr": opt.param_groups[0]["lr"]})
            if val_auc > best_auc:
                best_auc, best_epoch = val_auc, epoch
                self._save(ckpt_path, model, loaders, val_auc, epoch)

        if best_epoch < 0:                                    # 0 epochs -> still emit a usable ckpt
            self._save(ckpt_path, model, loaders, best_auc, best_epoch)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"best_val_auc": best_auc, "best_epoch": best_epoch,
                       "config": asdict(cfg), "history": history}, f, indent=2)
        return ckpt_path

    @torch.no_grad()
    def _val_auc(self, model, loader, device) -> float:
        model.eval()
        ys, ps = [], []
        for xb, yb in loader:
            logits = model(xb.to(device))
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(yb.numpy())
        if not ys:
            return float("nan")
        return float(M.auc(np.concatenate(ys), np.concatenate(ps)))

    def _save(self, path, model, loaders, val_auc, epoch):
        torch.save({
            "detector": getattr(self.det.spec, "name", type(self.det).__name__),
            "state_dict": model.state_dict(),
            "config": asdict(self.cfg),
            "feature_cols": loaders.feature_cols,
            "preproc": {"median": loaders.preproc.median, "mean": loaders.preproc.mean,
                        "std": loaders.preproc.std},
            "val_auc": float(val_auc), "epoch": int(epoch),
            "label": "generated-positive (y = 1 - is_real)",
        }, path)
