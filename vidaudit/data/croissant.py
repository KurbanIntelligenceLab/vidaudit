"""Emit ML Croissant (1.0) metadata for the released artifact.

What VidAudit redistributes is the per-clip *feature tables* + LOGO/RvR splits +
provenance labels (not the source videos). Croissant makes that artifact
machine-readable and loadable by standard tooling. This describes the feature
table's schema (provenance columns + a note that feature columns are
detector-specific) and links the canonical-pipeline recipe ids as provenance.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

# The standard Croissant 1.0 JSON-LD context (mlcommons.org/croissant/1.0).
CROISSANT_CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract",
    "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
    "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
    "repeated": "cr:repeated", "replace": "cr:replace", "sc": "https://schema.org/",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField",
    "transform": "cr:transform",
}

_DTYPE = {"video_id": "sc:Text", "generator": "sc:Text", "dataset": "sc:Text",
          "label": "sc:Text", "is_real": "sc:Integer"}


def _field(file_id: str, col: str) -> Dict:
    return {"@type": "cr:Field", "@id": f"clips/{col}", "name": col,
            "dataType": _DTYPE.get(col, "sc:Float"),
            "source": {"fileObject": {"@id": file_id}, "extract": {"column": col}}}


def feature_table_croissant(
    name: str, *, description: str, feature_files: Sequence[Dict],
    meta_columns: Sequence[str] = ("video_id", "generator", "is_real"),
    version: str = "0.1.0", license: str = "", homepage: str = "",
    cite_as: str = "", recipe: Optional[Dict] = None,
) -> Dict:
    """Build a Croissant record for the released feature tables.

    `feature_files`: dicts like {"id","name","contentUrl"[,"sha256"]} (one per CSV).
    `recipe`: optional provenance, e.g. {"canonical": "h264-...", "kfilter": {...}}.
    """
    if not feature_files:
        raise ValueError("feature_files must list at least one feature CSV")
    primary = feature_files[0]["id"]

    distribution = []
    for f in feature_files:
        obj = {"@type": "cr:FileObject", "@id": f["id"], "name": f.get("name", f["id"]),
               "contentUrl": f.get("contentUrl", f["id"]),
               "encodingFormat": f.get("encodingFormat", "text/csv")}
        if f.get("sha256"):
            obj["sha256"] = f["sha256"]
        distribution.append(obj)

    record = {
        "@context": CROISSANT_CONTEXT, "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": name, "description": description, "version": version,
        "distribution": distribution,
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "clips", "name": "clips",
            "description": "Per-clip provenance; detector feature columns vary by table "
                           "(all numeric columns outside the provenance fields).",
            "key": {"@id": "clips/video_id"},
            "field": [_field(primary, c) for c in meta_columns],
        }],
    }
    if license:
        record["license"] = license
    if homepage:
        record["url"] = homepage
    if cite_as:
        record["citeAs"] = cite_as
    if recipe:
        record["dct:provenance"] = recipe
    return record


def write_croissant(path: str, *args, **kwargs) -> str:
    """Build (via feature_table_croissant) and write a croissant.json; returns the path."""
    record = feature_table_croissant(*args, **kwargs)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
