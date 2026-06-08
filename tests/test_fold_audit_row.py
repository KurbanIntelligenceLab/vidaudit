"""fold_audit_row updates the matching leaderboard row in place from an audit JSON: it fills
the audited metric columns, flips the row off `pending`, refuses to invent a row for an
unknown model (published rows are authoritative), and rewrites ONLY the matched line so other
rows (incl. `n/a` cells and hand-authored quoting) stay byte-identical."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fold_audit_row.py"
_spec = importlib.util.spec_from_file_location("fold_audit_row", _SCRIPT)
fold_audit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fold_audit_row)

_HEADER = ("model,ours,family,backbone,weights,benchmark,cell,readout,logo_id,logo_ood,rvr,"
           "pauc10,tpr1,tpr01,brier,ece,cost_ms,status,paper,source,notes")
# a pending target row, an authoritative published row, and a "tricky" row whose weights cell
# is the literal `n/a` and whose notes contain only a semicolon (NaN-token + quoting traps).
_AIGVDET = ("AIGVDet,0,fusion,two-stream ResNet50,public,GenVidBench,matched-27k,native-head,"
            ",,,,,,,,,pending,paper,,notes")
_WAVEREP = ("WaveRep,0,forensic,DINOv2,public,GenVidBench,matched-27k,native-LLR,"
            "0.996,0.996,0.534,0.98,0.935,0.816,0.026,0.039,420,,paper,src,")
_PROBE = ('clip-length probe,1,artifact,clip length,n/a,GenVidBench,matched-27k,L2-LR,'
          ',0.998,,,,,,,,leakage,ours,Paper/Results,"0.998 unaudited; falls under P2"')


def _csv(tmp_path) -> str:
    p = tmp_path / "leaderboard.csv"
    # CRLF, like the real leaderboard.csv, so the byte-identity test catches CRLF->LF churn.
    p.write_bytes(("\r\n".join([_HEADER, _AIGVDET, _WAVEREP, _PROBE]) + "\r\n").encode())
    return str(p)


def _audit(tmp_path) -> str:
    p = tmp_path / "aigvdet.audit.json"
    p.write_text(json.dumps({
        "readout": "native-score", "logo_id": 0.812, "logo_ood": 0.733,
        "rvr": 0.598, "margin": 0.135, "pauc10": 0.61, "tpr1": 0.22,
        "tpr01": 0.05, "brier": 0.19, "ece": 0.11,
        "floor_verdict": "certified", "deploy_tier": "collapses",
    }))
    return str(p)


def _row(csv_path, model):
    for ln in Path(csv_path).read_text().splitlines()[1:]:
        if ln.split(",", 1)[0] == model:
            return ln
    raise AssertionError(f"row {model} not found")


def test_fold_fills_metrics_and_clears_pending(tmp_path):
    csv = _csv(tmp_path)
    fold_audit_row.fold("AIGVDet", _audit(tmp_path), csv_path=csv, render=False)
    cells = _row(csv, "AIGVDet").split(",")
    # header positions: logo_ood=9, rvr=10, tpr01=13, status=17, source=19
    assert cells[9] == "0.733" and cells[10] == "0.598" and cells[13] == "0.05"
    assert cells[17] == ""                                  # flipped off pending
    assert cells[19] == "aigvdet.audit.json"


def test_fold_preserves_untouched_rows_byte_identical(tmp_path):
    csv = _csv(tmp_path)
    fold_audit_row.fold("AIGVDet", _audit(tmp_path), csv_path=csv, render=False)
    raw = Path(csv).read_bytes()
    # untouched rows present verbatim WITH their CRLF terminator: catches `n/a` blanking,
    # quoting drift, AND CRLF->LF churn (the published row + the n/a/semicolon row).
    assert (_WAVEREP + "\r\n").encode() in raw
    assert (_PROBE + "\r\n").encode() in raw
    # no LF-only line ending introduced anywhere (the whole file stays CRLF)
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_fold_refuses_unknown_model(tmp_path):
    with pytest.raises(SystemExit, match="no leaderboard row"):
        fold_audit_row.fold("NotAModel", _audit(tmp_path), csv_path=_csv(tmp_path), render=False)
