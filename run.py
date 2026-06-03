#!/usr/bin/env python3
"""VidAudit command-line entry point.

    python run.py leaderboard                      # render LEADERBOARD.md from leaderboard.csv
    python run.py eval --features clips.csv        # audit a precomputed feature table
    python run.py eval <model>                     # audit a registered detector from clips [in progress]
    python run.py train <model>                    # train a method with a recipe         [in progress]
    python run.py fetch-weights <name>             # download + verify weights            [planned]
    python run.py fetch-data <name>                # fetch a dataset package              [planned]
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

    ev = sub.add_parser("eval", help="audit a detector, or a precomputed feature table via --features")
    ev.add_argument("model", nargs="?", default=None, help="registered detector name (clip eval: in progress)")
    ev.add_argument("--features", help="precomputed per-clip feature CSV to audit")
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

    tr = sub.add_parser("train", help="train a detector that ships a recipe [in progress]")
    tr.add_argument("model")

    for name in ("fetch-weights", "fetch-data"):
        s = sub.add_parser(name, help=f"{name.replace('-', ' ')} (download + verify) [planned]")
        s.add_argument("name")

    args = p.parse_args(argv)

    if args.cmd == "leaderboard":
        from vidaudit.audit.leaderboard import build
        print(f"rendered {build(args.csv, args.out)}")
        return 0

    if args.cmd == "extract":
        from vidaudit.detectors._extract import clips_from_manifest, extract_table
        from vidaudit.detectors.registry import get
        det = get(args.model)
        df = extract_table(det, clips_from_manifest(args.manifest), kind=args.kind, out=args.out)
        print(f"extracted {len(df)} rows -> {args.out}")
        return 0

    if args.cmd == "eval":
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

    if args.cmd in ("train", "fetch-weights", "fetch-data"):
        print(f"`{args.cmd}` is not wired yet (see the roadmap in README.md). "
              f"Live now: `python run.py leaderboard` and `python run.py eval --features <csv>`.",
              file=sys.stderr)
        return 2

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
