"""Model-zoo weight fetching: download + sha256-verify + cache, driven by zoo/manifest.yaml.

A detector's load_weights() calls fetch_weights(name) to get a local path to its
checkpoint (downloaded from the URL in the manifest and verified against the recorded
sha256, cached under ~/.cache/vidaudit/weights). Published checkpoints are hosted on
funded storage; until a URL is filled in the manifest, fetch_weights raises a clear
message and the user can pass a local --weights path instead.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "zoo" / "manifest.yaml"
CACHE = Path(os.environ.get("VIDAUDIT_WEIGHTS", Path.home() / ".cache" / "vidaudit" / "weights"))

_PLACEHOLDER = "PLACEHOLDER"


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(name: str, *, manifest: dict | None = None, force: bool = False) -> Path:
    """Return a local path to `name`'s checkpoint, downloading + verifying if needed."""
    entry = (manifest or load_manifest()).get(name.lower())
    if not entry:
        raise KeyError(f"no zoo/manifest.yaml entry for {name!r}")
    w = entry["weights"][0]
    url, sha, fn = w.get("url", ""), w.get("sha256"), w.get("file", f"{name.lower()}.ckpt")
    dest = CACHE / name.lower() / fn
    if dest.exists() and not force:
        return dest
    if not url or not url.lower().startswith(("http://", "https://")):
        raise RuntimeError(
            f"{name}: no public download URL in zoo/manifest.yaml (the checkpoint is on "
            f"cluster permanent storage; see the entry's `url` note). rsync it locally and "
            f"pass the path: load_weights('/path/to/{fn}') or "
            f"`run.py extract {name.lower()} --weights /path/to/{fn}`."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[zoo] fetching {name} weights -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    if sha and _sha256(dest) != sha:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: sha256 mismatch after download (expected {sha})")
    return dest
