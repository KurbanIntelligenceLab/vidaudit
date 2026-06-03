"""Seed the audited leaderboard CSV from the WACV 2027 audit results.

Each row is a verified audited result; matched 27k GenVidBench cell unless the
`cell` field says otherwise. Provenance is in `source` (files under Paper/Results
on the cluster). Re-run to regenerate ../leaderboard.csv. The leaderboard renderer
derives the verdicts from these numbers, so nothing here asserts a verdict except
the special `status` rows (leakage existence proof, excluded methods).

    python scripts/seed_leaderboard.py
"""
from __future__ import annotations

import csv
from pathlib import Path

FIELDS = [
    "model", "ours", "family", "backbone", "weights", "benchmark", "cell", "readout",
    "logo_id", "logo_ood", "rvr", "pauc10", "tpr1", "tpr01", "brier", "ece", "cost_ms",
    "status", "paper", "source", "notes",
]

# matched 27k GenVidBench cell (leave-one-generator-out), unless noted
ROWS = [
    dict(model="WaveRep", ours=0, family="forensic", backbone="DINOv2 ViT-B/14 + wavelet aug",
         weights="public", readout="native-LLR",
         logo_id=0.996, logo_ood=0.996, rvr=0.534, pauc10=0.980, tpr1=0.935, tpr01=0.816,
         brier=0.026, ece=0.039, cost_ms=420,
         paper="Corvi et al. 2025, NeurIPS (arXiv:2506.16802)",
         source="waverep_native_readout.json"),
    dict(model="XSFF", ours=1, family="fusion", backbone="codec MV + DINOv2/ReStraV (34-d)",
         weights="ours", readout="L2-LR",
         logo_id=0.996, logo_ood=0.946, rvr=0.604, cost_ms=24,
         paper="ours (WACV 2027)", source="Paper/Results", notes="MV+ReStraV fusion"),
    dict(model="ReStraV", ours=0, family="appearance", backbone="DINOv2 ViT-S/14 (21-d)",
         weights="public", readout="L2-LR",
         logo_id=0.995, logo_ood=0.931, rvr=0.586, pauc10=0.871, tpr1=0.726, tpr01=0.634,
         brier=0.097, ece=0.120, cost_ms=10,
         paper="Interno et al. 2025, NeurIPS (arXiv:2507.00583)",
         source="restrav_native_head.md",
         notes="published balanced AUC 0.996; native MLP head 0.901 at RvR 0.633"),
    dict(model="D3", ours=0, family="appearance", backbone="XCLIP-ViT-B/16",
         weights="training-free", readout="native-head",
         logo_id=0.887, logo_ood=0.887, rvr=0.421, cost_ms=50,
         paper="Zheng et al. 2025, ICCV (arXiv:2508.00701)",
         source="_aigvd_d3_native", notes="training-free, so ID equals OOD"),
    dict(model="FVMD", ours=0, family="motion", backbone="PIPs++ point tracker (1024-d)",
         weights="metric", readout="L2-LR",
         logo_id=0.924, logo_ood=0.880, rvr=0.574, pauc10=0.705, tpr1=0.107, tpr01=0.027,
         brier=0.127, ece=0.087, cost_ms=59,
         paper="FVMD (Liu et al. 2024)", source="Paper/Results"),
    dict(model="TemporalSpec+aug", ours=1, family="codec", backbone="codec motion vectors (20-d)",
         weights="ours", readout="LightGBM",
         logo_id=0.960, logo_ood=0.871, rvr=0.634, cost_ms=15,
         paper="ours (WACV 2027)", source="Paper/Results"),
    dict(model="RAFT", ours=0, family="motion", backbone="RAFT-Large optical flow (T0+T1)",
         weights="pretrained", readout="L2-LR",
         logo_id=0.901, logo_ood=0.855, rvr=0.627, pauc10=0.650, tpr1=0.110, tpr01=0.020,
         brier=0.158, ece=0.104, cost_ms=400,
         paper="RAFT (Teed and Deng 2020)", source="Paper/Results"),
    dict(model="CLIP", ours=0, family="appearance", backbone="CLIP-ViT-B/32",
         weights="pretrained", readout="L2-LR",
         logo_id=0.945, logo_ood=0.852, rvr=0.766, pauc10=0.782, tpr1=0.417, tpr01=0.238,
         brier=0.128, ece=0.094, cost_ms=67,
         paper="CLIP (Radford et al. 2021)", source="Paper/Results",
         notes="rides dataset identity (high RvR)"),
    dict(model="TemporalSpec", ours=1, family="codec", backbone="codec motion vectors (13-d)",
         weights="ours", readout="L2-LR",
         logo_id=0.909, logo_ood=0.832, rvr=0.643, pauc10=0.665, tpr1=0.130, tpr01=0.024,
         brier=0.166, ece=0.106, cost_ms=14,
         paper="ours (WACV 2027)", source="Paper/Results", notes="white-box control"),
    dict(model="NSG-VD", ours=0, family="forensic", backbone="diffusion noise-score (2-d)",
         weights="public-ckpts", readout="L2-LR",
         logo_id=0.696, logo_ood=0.660, rvr=0.596, cost_ms="",
         paper="Zhang et al. 2025, NeurIPS (arXiv:2510.08073)",
         source="nsgvd_readout_sweep.json",
         notes="best readout (lr); published-weights method that nearly collapses under the audit"),
    # existence proof (special row)
    dict(model="clip-length probe", ours=1, family="artifact",
         backbone="clip length / non-I-frame count (3-d)", weights="n/a", readout="L2-LR",
         logo_ood=0.998, status="leakage",
         paper="ours (existence proof)", source="Paper/Results",
         notes="0.998 unaudited, falls to 0.529 under the P2 K-frame filter"),
    # other cells
    dict(model="TemporalSpec", ours=1, family="codec", backbone="codec motion vectors (13-d)",
         weights="ours", benchmark="GenVidBench", cell="full-116k", readout="L2-LR",
         logo_id=0.904, logo_ood=0.819, rvr=0.628,
         paper="ours (WACV 2027)", source="Paper/Results", notes="full unmatched cell"),
    dict(model="D3", ours=0, family="appearance", backbone="XCLIP-ViT-B/16",
         weights="training-free", benchmark="AIGVDBench", cell="aigvd-2284", readout="native-head",
         logo_id=0.771, logo_ood=0.771, cost_ms=50,
         paper="Zheng et al. 2025, ICCV (arXiv:2508.00701)", source="_aigvd_d3_native",
         notes="2025 generators (Wan2.1 0.659); RvR n/a (one real source); run used B/32, re-run at B/16"),
    # excluded
    dict(model="DeMamba", ours=0, family="appearance", backbone="Mamba SSM",
         weights="withheld", status="excluded",
         paper="Chen et al. 2024 (arXiv:2405.19707)", source="",
         notes="pretrained weights withheld (GitHub issues #5/#16/#21); cannot re-score"),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "leaderboard.csv"
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in ROWS:
            row = {k: r.get(k, "") for k in FIELDS}
            row.setdefault("benchmark", "GenVidBench")
            if not row["benchmark"]:
                row["benchmark"] = "GenVidBench"
            if not row["cell"]:
                row["cell"] = "matched-27k"
            w.writerow(row)
            n += 1
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
