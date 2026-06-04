"""STALL: a training-free AI-generated video detector (Ben Hayun et al., CVPR 2026;
arXiv:2603.15026; github.com/OmerBenHayun/STALL). A frozen DINOv3 ViT-L/16 embeds frames; a
clip is scored by whitened spatial (MAX) and temporal (MIN) Gaussian log-likelihoods, each
mapped to a percentile against a real-video (VATEX) calibration set. The final score is the
mean of the two percentiles (higher = more real), so score() returns 1 - final.

The upstream repo carries no license, so the scoring math is reimplemented from the paper.
The released calibration npz and the DINOv3 backbone (Meta DINOv3 License) are obtained from
their original sources.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_LOG2PI = float(np.log(2.0 * np.pi))
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- pure, testable scoring math (no backbone) -----------------------------
def whiten(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply a pre-fit whitening transform: (X - mu) @ W. X is [n, d]."""
    return (np.asarray(X, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) @ np.asarray(W, dtype=np.float64)


def gaussian_loglik(Y: np.ndarray) -> np.ndarray:
    """Per-row log-likelihood under an isotropic standard normal N(0, I_d).

    l(y) = -0.5 * (d * log(2*pi) + ||y||_2^2). Returns one scalar per row of Y [n, d]."""
    Y = np.asarray(Y, dtype=np.float64)
    d = Y.shape[-1]
    return -0.5 * (d * _LOG2PI + np.sum(Y * Y, axis=-1))


def percentile_le(score: float, calib: np.ndarray) -> float:
    """Fraction of the (real) calibration scores <= `score`, in [0, 1]."""
    calib = np.asarray(calib, dtype=np.float64)
    if calib.size == 0:
        return 0.5
    return float(np.mean(calib <= score))


def stall_scores(emb: np.ndarray, params: dict, eps: float = 1e-8) -> Tuple[float, float, float, float, float]:
    """Score a clip from its frame embeddings + the calibration params.

    `emb` is [T, d] (T frame embeddings). `params` holds the pre-fit spatial/temporal
    whitening (mu/W) and the real calibration log-likelihoods. Returns
    (spat_pct, temp_pct, spat_raw_maxLL, temp_raw_minLL, final) where final = mean of the
    two percentiles in [0, 1] (higher = more real)."""
    emb = np.asarray(emb, dtype=np.float64)

    # spatial branch: whitened per-frame likelihood, aggregated with MAX
    ys = whiten(emb, params["mu_spat"], params["W_spat"])
    spat_raw = float(np.max(gaussian_loglik(ys)))

    # temporal branch: L2-normalized consecutive diffs, whitened, aggregated with MIN
    diffs = emb[1:] - emb[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    nz = norms > eps
    unit = np.zeros_like(diffs)
    unit[nz] = diffs[nz] / norms[nz, None]
    yt = whiten(unit, params["mu_temp"], params["W_temp"])
    llt = gaussian_loglik(yt)
    llt[~nz] = np.inf                       # zero-diff frames ignored by the MIN
    temp_raw = float(np.min(llt)) if np.any(nz) else float("inf")

    spat_pct = percentile_le(spat_raw, params["calib_spat"])
    temp_pct = percentile_le(temp_raw, params["calib_temp"])
    final = 0.5 * (spat_pct + temp_pct)
    return spat_pct, temp_pct, spat_raw, temp_raw, final


# ---- npz param loading (flexible key names) --------------------------------
_PARAM_KEYS = {
    "mu_spat": ("mu_spat", "mu_spatial", "spat_mu"),
    "W_spat": ("W_spat", "W_spatial", "spat_W"),
    "mu_temp": ("mu_temp", "mu_temporal", "temp_mu"),
    "W_temp": ("W_temp", "W_temporal", "temp_W"),
    "calib_spat": ("calib_spat", "calib_ll_spat", "calib_spatial", "ref_spat"),
    "calib_temp": ("calib_temp", "calib_ll_temp", "calib_temporal", "ref_temp"),
}


def load_params(npz_path: str) -> dict:
    """Load the released STALL calibration `.npz` into the param dict `stall_scores` wants.

    Accepts a few candidate key spellings so a wrapper does not hard-fail on a minor
    naming difference in the released file."""
    raw = np.load(npz_path)
    out = {}
    for canonical, candidates in _PARAM_KEYS.items():
        hit = next((c for c in candidates if c in raw.files), None)
        if hit is None:
            raise KeyError(f"{npz_path}: missing param for {canonical!r}; "
                           f"file has {list(raw.files)}")
        out[canonical] = np.asarray(raw[hit])
    return out


@register("stall")
class STALL(Detector):
    spec = DetectorSpec(
        name="STALL", backbone="DINOv3 ViT-L/16 (frozen) + whitened Gaussian likelihood",
        family="fusion", published_weights=True, trainable=False, needs_gpu=True,
        weights_url="STALL calibration npz: https://github.com/OmerBenHayun/STALL ; "
                    "DINOv3 backbone (gated): https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
        license="STALL code/params undeclared (all-rights-reserved by default); DINOv3 "
                "backbone under the Meta DINOv3 License (gated). Scoring math reimplemented; "
                "params + backbone are point-to-source.",
        paper="Ben Hayun et al., CVPR 2026 (arXiv:2603.15026)",
        notes="Training-free; frozen DINOv3 ViT-L/16 + whitened spatial/temporal Gaussian "
              "likelihood vs VATEX calibration. score() = 1 - mean percentile. "
              "load_weights(<stall_params_vatex_dino_v3.npz>).",
    )
    feature_names = ["spat_pct", "temp_pct", "spat_ll", "temp_ll"]

    def __init__(self, n_frames: int = 16, dinov3_dir: Optional[str] = None,
                 dinov3_weights: Optional[str] = None, device=None):
        self.n_frames = n_frames
        self.dinov3_dir = dinov3_dir          # local clone of facebookresearch/dinov3
        self.dinov3_weights = dinov3_weights  # gated ViT-L/16 checkpoint
        self._device = device
        self._params: Optional[dict] = None
        self._dino = None

    def _dev(self):
        import torch
        if self._device is None:
            self._device = ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    def load_weights(self, path: Optional[str] = None) -> None:
        """Load the released STALL calibration `.npz` (point-to-source from the STALL repo).

        The gated DINOv3 backbone is supplied separately via the constructor
        (`dinov3_dir` + `dinov3_weights`) since it cannot be auto-downloaded."""
        if not path:
            raise ValueError("STALL needs the released calibration params: "
                             "load_weights('<stall_params_vatex_dino_v3.npz>'). Obtain it "
                             "from the STALL repo; DINOv3 (gated) is set on the constructor.")
        self._params = load_params(path)

    def _ensure_dino(self):
        """Lazily load the frozen DINOv3 ViT-L/16 (gated; user-provided)."""
        if self._dino is None:
            import torch
            if not self.dinov3_dir:
                raise RuntimeError("DINOv3 is gated and not auto-downloadable. Clone "
                                   "facebookresearch/dinov3 and pass STALL(dinov3_dir=..., "
                                   "dinov3_weights=<vitl16 .pth>).")
            self._dino = torch.hub.load(self.dinov3_dir, "dinov3_vitl16", source="local",
                                        weights=self.dinov3_weights).eval().to(self._dev())
        return self._dino

    def _embed(self, clip: Clip) -> np.ndarray:
        """Frozen DINOv3 CLS embedding per frame -> [T, 1024]."""
        import torch
        from torchvision import transforms
        from vidaudit.detectors._extract import decode_pil
        model = self._ensure_dino()
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
        frames = decode_pil(clip.path, n_frames=self.n_frames)
        batch = torch.stack([tf(im.convert("RGB")) for im in frames]).to(self._dev())
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, dict):       # some hub variants return a dict of tokens
                out = out.get("x_norm_clstoken", next(iter(out.values())))
        return out.float().cpu().numpy()

    def _require_params(self) -> dict:
        if self._params is None:
            raise RuntimeError("STALL calibration params not loaded; call "
                               "load_weights('<stall_params_vatex_dino_v3.npz>') first.")
        return self._params

    def score(self, clip: Clip) -> float:
        """p(generated) = 1 - final STALL score (final is HIGHER = more real)."""
        _, _, _, _, final = stall_scores(self._embed(clip), self._require_params())
        return float(1.0 - final)

    def features(self, clip: Clip) -> np.ndarray:
        """The method's interpretable evidence: [spatial %, temporal %, spatial LL, temporal LL]."""
        spat_pct, temp_pct, spat_raw, temp_raw, _ = stall_scores(self._embed(clip), self._require_params())
        return np.array([spat_pct, temp_pct, spat_raw, temp_raw], dtype=np.float64)
