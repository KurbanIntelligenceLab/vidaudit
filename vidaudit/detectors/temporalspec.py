"""TemporalSpec (ours, WACV 2027): the white-box codec-motion-vector control.

Extracts H.264 motion vectors directly via PyAV (the `export_mvs` codec flag), bins
them into a 16x16 macroblock grid [T, H_mb, W_mb, 5] with channels
[SOURCE, DST_X, DST_Y, SRC_X, SRC_Y] (matching the cluster's MotionVectorExtractor),
and runs the shared 13-d spectral extractor (vidaudit.features). No neural network and
no external tools: PyAV is already in the environment, so this is fully clone-and-run.
The codec-MV machinery is contained entirely in this module, so the other detectors are
unaffected.
"""
from __future__ import annotations

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register
from vidaudit.features import FEATURE_NAMES, feature_vector


@register("temporalspec")
class TemporalSpec(Detector):
    spec = DetectorSpec(
        name="TemporalSpec", backbone="codec motion vectors (13-d)", family="codec",
        published_weights=False, trainable=False, needs_gpu=False, cost_ms=14.0,
        paper="ours (WACV 2027)",
        notes="white-box control; H.264 motion vectors via PyAV export_mvs -> 16x16 "
              "macroblock grid -> shared 13-d spectral extractor. No NN, no external tools.",
    )
    feature_names = FEATURE_NAMES

    def __init__(self, target_frames=None, mb: int = 16):
        self.target_frames = target_frames
        self.mb = mb

    def _extract_mv(self, path) -> np.ndarray:
        """Decode H.264 motion vectors -> [T, H_mb, W_mb, 5] macroblock array.

        Mirrors MotionVectorExtractor.parse: keep 16x16 macroblocks, bin by source
        position, first MV wins per cell; channels [SOURCE, DST_X, DST_Y, SRC_X, SRC_Y].
        """
        import av
        from av.codec.context import Flags2
        mb = self.mb
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.codec_context.flags2 |= Flags2.export_mvs
            dim_x = dim_y = None
            grids = []
            for frame in container.decode(stream):
                if dim_x is None:
                    dim_x = (frame.width - 1) // mb + 1
                    dim_y = (frame.height - 1) // mb + 1
                grid = np.zeros((dim_y, dim_x, 5), dtype=np.int64)
                try:
                    mvs = frame.side_data.get("MOTION_VECTORS")
                except Exception:
                    mvs = None
                if mvs is not None:
                    for r in mvs.to_ndarray():
                        if r["w"] == mb and r["h"] == mb:
                            x = int(r["src_x"]) // mb
                            y = int(r["src_y"]) // mb
                            if 0 <= x < dim_x and 0 <= y < dim_y and not grid[y, x].any():
                                grid[y, x] = (r["source"], r["dst_x"], r["dst_y"],
                                              r["src_x"], r["src_y"])
                grids.append(grid)
        finally:
            container.close()
        if not grids:
            raise RuntimeError(f"no frames decoded: {path}")
        return np.stack(grids, axis=0)

    def features(self, clip: Clip) -> np.ndarray:
        mv = self._extract_mv(clip.path)
        v = feature_vector(mv, target_frames=self.target_frames)
        if v is None:
            raise RuntimeError("extract_features returned None (too few motion frames)")
        return v
