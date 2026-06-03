#!/usr/bin/env python3
"""VidAudit command-line entry point.

    python run.py leaderboard                 # render LEADERBOARD.md from leaderboard.csv
    python run.py eval <model> [--cell ...]   # audit a registered detector  [in progress]
    python run.py train <model>               # train a method with a recipe [in progress]
    python run.py fetch-weights <name>        # download + verify weights     [planned]
    python run.py fetch-data <name>           # fetch a dataset package       [planned]
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

    ev = sub.add_parser("eval", help="run a registered detector through the P1-P6 audit [in progress]")
    ev.add_argument("model")
    ev.add_argument("--cell", default="genvidbench27k")

    tr = sub.add_parser("train", help="train a detector that ships a recipe [in progress]")
    tr.add_argument("model")

    for name in ("fetch-weights", "fetch-data"):
        s = sub.add_parser(name, help=f"{name.replace('-', ' ')} (download + verify) [planned]")
        s.add_argument("name")

    args = p.parse_args(argv)

    if args.cmd == "leaderboard":
        from vidaudit.audit.leaderboard import build
        out = build(args.csv, args.out)
        print(f"rendered {out}")
        return 0

    if args.cmd in ("eval", "train", "fetch-weights", "fetch-data"):
        print(
            f"`{args.cmd}` is not wired yet (see the roadmap in README.md). "
            f"The audited leaderboard pipeline is live: try `python run.py leaderboard`.",
            file=sys.stderr,
        )
        return 2

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
