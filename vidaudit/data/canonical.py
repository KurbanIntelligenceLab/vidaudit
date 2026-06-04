"""P1 - canonical re-encode.

Generators leave codec/container fingerprints (bitrate ladders, GOP structure,
chroma subsampling, mux quirks) that a detector can exploit instead of learning
real-vs-generated content. P1 pushes every clip - real and generated - through one
fixed H.264 recipe so those fingerprints are normalized away before any feature is
extracted. H.264 specifically: the codec motion-vector side data the codec-MV
detectors read is H.264-only (an HEVC re-encode yields all-zero MV arrays).

`CanonicalSpec` is the recipe as data (recorded in provenance via `recipe_id`);
`ffmpeg_cmd` builds the exact argv (transparent + reproducible); `canonicalize`
runs it. Re-encoding a full dataset is a cluster job, not a laptop job.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass(slots=True, frozen=True)
class CanonicalSpec:
    vcodec: str = "libx264"
    crf: int = 23                     # constant quality; fixed so it cannot leak
    preset: str = "medium"
    pix_fmt: str = "yuv420p"          # 4:2:0, the common denominator
    gop: int = 16                     # closed, fixed GOP -> deterministic MV structure
    fps: Optional[float] = None       # None = keep source rate
    max_side: Optional[int] = None    # None = keep resolution; else downscale longest side
    drop_audio: bool = True
    faststart: bool = True


CANONICAL = CanonicalSpec()           # the released default recipe


def recipe_id(spec: CanonicalSpec = CANONICAL) -> str:
    """Short stable id of a recipe, for provenance (same spec -> same id)."""
    payload = json.dumps(asdict(spec), sort_keys=True)
    return "h264-" + hashlib.sha1(payload.encode()).hexdigest()[:12]


def ffmpeg_cmd(src: str, dst: str, spec: CanonicalSpec = CANONICAL) -> List[str]:
    """The exact ffmpeg argv for the canonical re-encode (deterministic GOP, no audio)."""
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", src,
           "-c:v", spec.vcodec, "-crf", str(spec.crf), "-preset", spec.preset,
           "-pix_fmt", spec.pix_fmt, "-g", str(spec.gop),
           "-x264-params", f"keyint={spec.gop}:min-keyint={spec.gop}:scenecut=0"]
    if spec.fps:
        cmd += ["-r", str(spec.fps)]
    if spec.max_side:
        # keep aspect, force even dims (yuv420p needs them); only downscale, never up
        cmd += ["-vf", f"scale='if(gt(iw,ih),min({spec.max_side},iw),-2)':"
                       f"'if(gt(iw,ih),-2,min({spec.max_side},ih))'"]
    if spec.drop_audio:
        cmd += ["-an"]
    if spec.faststart:
        cmd += ["-movflags", "+faststart"]
    return cmd + [dst]


def canonicalize(src: str, dst: str, spec: CanonicalSpec = CANONICAL) -> str:
    """Re-encode `src` -> `dst` under the canonical recipe; returns `dst`.

    Raises a clear error if ffmpeg is absent (the conda env ships it). Skips work if
    `dst` already exists (idempotent, so reconstruction is resumable)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; it ships with the vidaudit conda env "
                           "(`conda activate vidaudit`).")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    proc = subprocess.run(ffmpeg_cmd(src, dst, spec), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"canonical re-encode failed for {src!r}:\n{tail}")
    return dst
