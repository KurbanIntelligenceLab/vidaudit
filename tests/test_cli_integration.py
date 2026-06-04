"""End-to-end CLI integration: drive run.main() through the analytical chain
(eval --features -> train -> eval --scores) on tiny synthetic tables. Unit tests
cover each module; this guards the CLI wiring (arg parsing, handler dispatch,
cross-module data shapes) that only breaks when the commands are chained."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import run


def _feature_table(path: str, d: int = 13, n: int = 60, shift: float = 1.7) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for src in ("real_a", "real_b"):
        for i in range(n):
            rows.append({"video_id": f"{src}_{i}", "generator": src, "is_real": 1,
                         **{f"f_{j}": float(rng.normal(0, 1)) for j in range(d)}})
    for g in ("g1", "g2", "g3"):
        for i in range(n):
            rows.append({"video_id": f"{g}_{i}", "generator": g, "is_real": 0,
                         **{f"f_{j}": float(rng.normal(shift, 1)) for j in range(d)}})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def test_cli_eval_features_then_train_then_eval_scores(tmp_path, capsys):
    feats = str(tmp_path / "feats.csv")
    df = _feature_table(feats)

    # 1) audit the feature table (retrains the L2-LR readout)
    assert run.main(["eval", "--features", feats]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["logo_ood"] > 0.8 and rec["n_features"] == 13

    # 2) train a head over the same table
    out_dir = str(tmp_path / "run")
    assert run.main(["train", "mlp-probe", "--features", feats, "--out", out_dir,
                     "--set", "epochs=10", "--set", "hidden=32"]) == 0
    assert os.path.exists(os.path.join(out_dir, "model.pt"))
    assert "trained mlp-probe" in capsys.readouterr().out      # flush before the next parse

    # 3) audit a fixed score column (native/trained-head path, no readout)
    scores = str(tmp_path / "scores.csv")
    rng = np.random.default_rng(1)
    df = df.assign(score=(1 - df["is_real"]) * 0.6 + rng.normal(0.2, 0.05, len(df)))
    df.to_csv(scores, index=False)
    assert run.main(["eval", "--scores", scores]) == 0
    rec2 = json.loads(capsys.readouterr().out)
    assert rec2["readout"] == "native-score" and rec2["logo_ood"] > 0.8


def test_cli_train_rejects_eval_only_detector(tmp_path, capsys):
    feats = str(tmp_path / "feats.csv")
    _feature_table(feats)
    # clip is a frozen reference (trainable=False) -> guarded, non-zero exit
    assert run.main(["train", "clip", "--features", feats]) == 2
    assert "eval-only" in capsys.readouterr().err
