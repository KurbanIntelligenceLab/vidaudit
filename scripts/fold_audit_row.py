#!/usr/bin/env python3
"""Fold a detector's audit JSON into leaderboard.csv, then re-render LEADERBOARD.md.

The audit JSON is the output of `run.py eval --scores ... > <model>.audit.json` (or
--features). This updates the EXISTING leaderboard row matched by model name (and optional
benchmark/cell), filling the audited metric columns and flipping the row off `pending`, so a
landed run becomes a real leaderboard row in one command instead of a hand edit.

    python scripts/fold_audit_row.py --model AIGVDet --audit /path/aigvdet.audit.json
    python scripts/fold_audit_row.py --model D3 --audit d3.json --benchmark GenVidBench --cell matched-27k

It does NOT invent a row: if no existing row matches, it errors (the published rows are
authoritative and must not be regenerated; only NEW models get vidaudit-computed rows). Only
the single matched line is rewritten; every other line is preserved byte-for-byte, so cells
like `n/a` and hand-authored quoting on untouched rows do not drift.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# leaderboard column <- audit-json key (only metric columns; static metadata is left as-is)
_METRIC_MAP = {
    "logo_id": "logo_id", "logo_ood": "logo_ood", "rvr": "rvr",
    "pauc10": "pauc10", "tpr1": "tpr1", "tpr01": "tpr01",
    "brier": "brier", "ece": "ece",
}


def _parse(line: str) -> list:
    return next(csv.reader([line]))


def fold(model: str, audit_path: str, csv_path: str | None = None,
         benchmark: str | None = None, cell: str | None = None,
         status: str = "", render: bool = True) -> int:
    """Update the single matching leaderboard line in place; return its line index."""
    csv_path = csv_path or str(_REPO / "leaderboard.csv")
    rec = json.loads(Path(audit_path).read_text())
    # newline="" keeps the original line endings untranslated (the leaderboard is CRLF); using
    # read_text()/write_text() would silently rewrite every line CRLF->LF and churn the diff.
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()
    if not lines:
        raise SystemExit(f"empty CSV: {csv_path}")

    header = _parse(lines[0].rstrip("\r\n"))
    col = {name: i for i, name in enumerate(header)}

    def field(fields: list, name: str) -> str:
        i = col.get(name)
        return fields[i] if i is not None and i < len(fields) else ""

    target = None
    for li in range(1, len(lines)):
        if not lines[li].strip():
            continue
        fields = _parse(lines[li].rstrip("\r\n"))
        if field(fields, "model").lower() != model.lower():
            continue
        if benchmark and field(fields, "benchmark").lower() != benchmark.lower():
            continue
        if cell and field(fields, "cell").lower() != cell.lower():
            continue
        if target is not None:
            raise SystemExit(f"multiple rows match model={model!r}; disambiguate with "
                             f"--benchmark / --cell.")
        target = (li, fields)

    if target is None:
        raise SystemExit(f"no leaderboard row for model={model!r} "
                         f"(benchmark={benchmark}, cell={cell}); refusing to invent one.")

    li, fields = target
    fields += [""] * (len(header) - len(fields))          # pad short rows to the header width
    for c, key in _METRIC_MAP.items():
        if c in col and rec.get(key) is not None:
            fields[col[c]] = str(rec[key])
    if "status" in col:
        fields[col["status"]] = status                    # "" = scored (off pending)
    if "source" in col:
        fields[col["source"]] = Path(audit_path).name

    # preserve THIS line's exact terminator so the only byte change is the row's data
    term = ("\r\n" if lines[li].endswith("\r\n")
            else "\n" if lines[li].endswith("\n")
            else "\r" if lines[li].endswith("\r") else "")
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)   # re-serialize ONLY this row
    lines[li] = buf.getvalue() + term
    with open(csv_path, "w", newline="") as f:            # all other lines stay byte-identical
        f.write("".join(lines))

    print(f"folded {model} <- {Path(audit_path).name}: "
          f"logo_ood={rec.get('logo_ood')} rvr={rec.get('rvr')} margin={rec.get('margin')} "
          f"tpr01={rec.get('tpr01')} floor={rec.get('floor_verdict')} "
          f"deploy={rec.get('deploy_tier')}")
    if render:
        from vidaudit.audit.leaderboard import build
        out = build(csv_path, str(_REPO / "LEADERBOARD.md"))
        print(f"rendered {out}")
    return li


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fold_audit_row", description=__doc__)
    p.add_argument("--model", required=True, help="leaderboard model name (e.g. AIGVDet)")
    p.add_argument("--audit", required=True, help="path to <model>.audit.json")
    p.add_argument("--csv", default=None, help="leaderboard CSV (default: repo leaderboard.csv)")
    p.add_argument("--benchmark", default=None, help="disambiguate when a model has >1 row")
    p.add_argument("--cell", default=None, help="disambiguate when a model has >1 row")
    p.add_argument("--status", default="", help="row status to set (default '' = scored)")
    p.add_argument("--no-render", action="store_true", help="skip re-rendering LEADERBOARD.md")
    a = p.parse_args(argv)
    fold(a.model, a.audit, a.csv, a.benchmark, a.cell, a.status, render=not a.no_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
