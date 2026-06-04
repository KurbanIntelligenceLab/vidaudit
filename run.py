#!/usr/bin/env python3
"""VidAudit command-line entry point.

    python run.py leaderboard                      # render LEADERBOARD.md from leaderboard.csv
    python run.py eval --features clips.csv        # audit a feature table (retrains the L2-LR readout)
    python run.py eval --scores clips.csv          # audit a fixed native/trained score (no readout)
    python run.py eval <model>                     # audit a registered detector from clips [in progress]
    python run.py train <model> --features f.csv   # train a method's head via the standard trainer
    python run.py prepare-data <ds> --source dir  # preprocess a downloaded dataset (P1 re-encode + P2 filter)
    python run.py fetch-weights <name>             # download + sha256-verify weights from the zoo
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="vidaudit",
        description="Audited evaluation + training toolkit for AI-generated video detection",
    )
    sub = p.add_subparsers(dest="cmd")

    lb = sub.add_parser("leaderboard", help="render the audited leaderboard from leaderboard.csv")
    lb.add_argument("--csv", default=None, help="input CSV (default: ./leaderboard.csv)")
    lb.add_argument("--out", default=None, help="output markdown (default: ./LEADERBOARD.md)")

    ev = sub.add_parser("eval", help="audit a feature table (--features) or a native/trained score (--scores)")
    ev.add_argument("model", nargs="?", default=None, help="registered detector name (clip eval: in progress)")
    ev.add_argument("--features", help="precomputed per-clip feature CSV to audit (retrains the L2-LR readout)")
    ev.add_argument("--scores", help="per-clip table with a fixed score column to audit (native/trained head, no readout)")
    ev.add_argument("--score-col", default="score", help="score column name for --scores (default: score)")
    ev.add_argument("--subset", default=None, help="matched-cell CSV with (video_id, generator)")
    ev.add_argument("--feature-cols", default=None,
                    help="explicit cols / 'auto:prefix=v,a'; default = all numeric except metadata")
    ev.add_argument("--reducer", default="none", choices=["none", "pca", "topk_l2"])
    ev.add_argument("--n-components", type=int, default=13)
    ev.add_argument("--classifier", default="lr", choices=["lr", "xgboost"])
    ev.add_argument("--inner-cv", action="store_true", help="5-fold inner CV over C (LR only)")
    ev.add_argument("--arrow", action="store_true", help="read the CSV with the pandas pyarrow backend (faster on wide tables)")

    ex = sub.add_parser("extract", help="extract a detector's features/score from clips into a CSV")
    ex.add_argument("model", help="registered detector name (e.g. d3)")
    ex.add_argument("--manifest", required=True,
                    help="clips CSV with (video_id, generator, label, is_real, mp4_path)")
    ex.add_argument("--out", required=True, help="output feature CSV (feeds `eval --features`)")
    ex.add_argument("--kind", default="auto", choices=["auto", "features", "score"])
    ex.add_argument("--weights", default=None,
                    help="local checkpoint path for a weighted detector (else fetched from the zoo)")
    ex.add_argument("--adm-ckpt", default=None,
                    help="NSG-VD only: path to the ADM diffusion model (256x256_diffusion_uncond.pt)")

    tr = sub.add_parser("train", help="train a detector's head via the standard trainer")
    tr.add_argument("model", help="a trainable detector (e.g. mlp-probe)")
    tr.add_argument("--features", help="training feature CSV (the `extract` output)")
    tr.add_argument("--val-features", default=None, help="held-out feature CSV (else split --features)")
    tr.add_argument("--subset", default=None, help="matched-cell CSV (video_id, generator)")
    tr.add_argument("--out", default="runs/train", help="output dir for the checkpoint + metrics")
    tr.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override any TrainConfig field (repeatable), e.g. --set lr=3e-4 --set epochs=100")

    fw = sub.add_parser("fetch-weights", help="download + sha256-verify weights from the zoo")
    fw.add_argument("name", help="zoo manifest entry (e.g. waverep, fvmd, nsgvd)")
    fw.add_argument("--force", action="store_true", help="re-download even if cached")

    pp = sub.add_parser("prepare-data",
                        help="preprocess a downloaded dataset (P1 re-encode + P2 filter) into the audit format")
    pp.add_argument("dataset", help="registered dataset adapter (genvidbench | aigvdbench)")
    pp.add_argument("--source", required=True, help="your local download of the ORIGINAL dataset")
    pp.add_argument("--out", required=True, help="output dir for canonical clips + the ready manifest")
    pp.add_argument("--min-frames", type=int, default=None, help="apply the P2 length filter with this minimum")
    pp.add_argument("--limit", type=int, default=None, help="cap total clips (dev runs)")
    pp.add_argument("--per-source", type=int, default=None, help="cap clips per source/generator (dev runs)")

    fd = sub.add_parser("fetch-data", help="low-level: reconstruct from a provenance manifest (P1 + P2)")
    fd.add_argument("--manifest", required=True,
                    help="provenance manifest CSV (video_id, generator, label, is_real, src_path|url)")
    fd.add_argument("--out", required=True, help="output dir for canonical clips + the ready manifest")
    fd.add_argument("--sources", default=None, help="dir of your downloaded source videos (for src_path rows)")
    fd.add_argument("--min-frames", type=int, default=None, help="apply the P2 length filter with this minimum")

    args = p.parse_args(argv)

    if args.cmd == "leaderboard":
        from vidaudit.audit.leaderboard import build
        print(f"rendered {build(args.csv, args.out)}")
        return 0

    if args.cmd == "extract":
        from vidaudit.detectors._extract import clips_from_manifest, extract_table
        from vidaudit.detectors.registry import get
        det = get(args.model)
        if args.weights:
            det.load_weights(args.weights)
        if getattr(args, "adm_ckpt", None) and hasattr(det, "adm_ckpt"):
            det.adm_ckpt = args.adm_ckpt        # NSG-VD's second checkpoint
        df = extract_table(det, clips_from_manifest(args.manifest), kind=args.kind, out=args.out)
        print(f"extracted {len(df)} rows -> {args.out}")
        return 0

    if args.cmd == "eval":
        if args.scores:
            import json
            from vidaudit.audit.protocol import audit_scores
            from vidaudit.data.cells import read_feature_table
            df = read_feature_table(args.scores, arrow=args.arrow)
            rec = audit_scores(df, args.score_col, subset=args.subset)
            rec.pop("per_generator", None)
            rec.pop("failure_audit", None)
            print(json.dumps(rec, indent=2))
            return 0
        if args.features:
            import json
            from vidaudit.audit.protocol import audit_features
            from vidaudit.data.cells import read_feature_table
            df = read_feature_table(args.features, arrow=args.arrow)
            rec = audit_features(
                df, args.feature_cols, subset=args.subset, reducer=args.reducer,
                n_components=args.n_components, classifier=args.classifier, inner_cv=args.inner_cv,
            )
            rec.pop("per_generator", None)
            rec.pop("failure_audit", None)
            print(json.dumps(rec, indent=2))
            return 0
        print("eval from raw clips needs a registered detector wrapper + the data package "
              "(in progress). To audit precomputed features now: "
              "`python run.py eval --features <csv>`.", file=sys.stderr)
        return 2

    if args.cmd == "train":
        from vidaudit.detectors.registry import get
        from vidaudit.train.config import apply_overrides
        det = get(args.model)
        if not det.is_trainable:
            print(f"`{args.model}` is eval-only (trainable=False): a training-free method "
                  f"or a frozen reference. Trainable detectors expose build_model().", file=sys.stderr)
            return 2
        cfg = det.default_train_config()
        if args.features:
            cfg.features = args.features
        if args.val_features:
            cfg.val_features = args.val_features
        if args.subset:
            cfg.subset = args.subset
        apply_overrides(cfg, args.overrides)
        if not cfg.features:
            print("training needs a feature table: `train <model> --features <extract.csv>` "
                  "(extract a backbone's features first).", file=sys.stderr)
            return 2
        out = det.train(cfg, [], args.out)
        print(f"trained {args.model} -> {out}")
        return 0

    if args.cmd == "fetch-weights":
        from vidaudit.zoo import fetch_weights
        try:
            print(f"weights -> {fetch_weights(args.name, force=args.force)}")
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            print(str(e), file=sys.stderr)
            return 2
        return 0

    if args.cmd == "prepare-data":
        from vidaudit.data.fetch import prepare_dataset
        from vidaudit.data.filters import KFilter
        kf = KFilter(min_frames=args.min_frames) if args.min_frames else None
        try:
            man = prepare_dataset(args.dataset, args.source, args.out, kfilter=kf,
                                  limit=args.limit, per_source=args.per_source)
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"prepared {args.dataset} -> {man}")
        return 0

    if args.cmd == "fetch-data":
        from vidaudit.data.fetch import reconstruct
        from vidaudit.data.filters import KFilter
        kf = KFilter(min_frames=args.min_frames) if args.min_frames else None
        try:
            man = reconstruct(args.manifest, args.out, sources_dir=args.sources, kfilter=kf)
        except (RuntimeError, FileNotFoundError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"reconstructed -> {man}")
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
