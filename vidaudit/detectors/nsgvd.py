"""NSG-VD (Zhang et al., NeurIPS 2025, arXiv:2510.08073): a diffusion-noise detector.

Per clip: decode 8 frames at 224x224, evaluate the ADM 256x256 unconditional diffusion
score at step 5/1000, build the Normalized Spatiotemporal Gradient (NSG) velocity feature
(score / pixel-difference), and pass the (1, 8, 3, 224, 224) NSG tensor through the trained
SingleSwinBlockDiscriminator -> a 300-d per-clip feature. `features(clip)` returns that
300-d vector. (The leaderboard's 0.660 comes from an MMD distance to a real reference bank,
a relative quantity that does not fit per-clip features(); use `mmd_scores` for that.)

Needs the ADM diffusion model + a per-generator Swin checkpoint (see zoo/manifest.yaml); the
NSG-VD codebase is vendored under vidaudit/_vendor/nsgvd and imported lazily. Ported from
run_nsgvd_features.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_VENDOR = Path(__file__).resolve().parents[1] / "_vendor" / "nsgvd"
_NUM_FRAMES = 8
_IMG = 224
_DIFFUSE_STEP = 5
_FEAT_DIM = 300


@register("nsgvd")
class NSGVD(Detector):
    spec = DetectorSpec(
        name="NSG-VD", backbone="ADM 256 diffusion + Swin discriminator", family="forensic",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="see zoo/manifest.yaml (nsgvd + nsgvd-adm); obtain from the original release",
        paper="Zhang et al., NeurIPS 2025 (arXiv:2510.08073)",
        notes="diffusion-noise NSG feature -> Swin 300-d; needs the ADM model + per-gen Swin "
              "ckpt; vendored codebase in vidaudit/_vendor/nsgvd (lazy).",
    )
    feature_names = [f"nsgvd_{j}" for j in range(_FEAT_DIM)]

    def __init__(self, adm_ckpt=None, disc_ckpt=None, n_frames: int = _NUM_FRAMES,
                 size: int = _IMG, diffuse_step: int = _DIFFUSE_STEP, device=None):
        self.adm_ckpt = adm_ckpt
        self._disc_ckpt = disc_ckpt
        self.n_frames = n_frames
        self.size = size
        self.diffuse_step = diffuse_step
        self._device = device
        self._model = None
        self._score_fn = None

    def load_weights(self, path=None) -> None:
        """`path` is the per-generator Swin discriminator checkpoint. The ADM diffusion
        model is set via `adm_ckpt` (constructor) or the 'nsgvd-adm' manifest entry."""
        if path:
            self._disc_ckpt = path
        self._model = None

    @staticmethod
    def _ensure_vendor():
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))

    def _dev(self):
        if self._device is None:
            import torch
            self._device = ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    def _resolve(self, which: str, attr):
        if attr:
            return str(attr)
        from vidaudit.zoo import fetch_weights
        return str(fetch_weights(which))

    def _build_discriminator(self):
        import torch
        self._ensure_vendor()
        from models.deep_mmd import deep_MMD
        from models.tall import SingleSwinBlockDiscriminator
        disc = SingleSwinBlockDiscriminator(num_features=_FEAT_DIM, duration=self.n_frames)
        model = deep_MMD(discriminator=disc, sigma=1000, sigma0=0.1, epsilon=10,
                         img_size=self.size, is_yy_zero=True, is_smooth=True)
        state = torch.load(self._resolve("nsgvd", self._disc_ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        return model.eval().to(self._dev())

    def _build_score_fn(self):
        """Build the ADM score predictor. The vendored get_score_fn loads
        ../Checkpoints/256x256_diffusion_uncond.pt relative to the nsgvd root, so we
        symlink the ADM checkpoint there and chdir during the call."""
        import torch  # noqa: F401
        self._ensure_vendor()
        adm = os.path.abspath(self._resolve("nsgvd-adm", self.adm_ckpt))
        ck_dir = _VENDOR.parent / "Checkpoints"
        ck_dir.mkdir(exist_ok=True)
        ck_target = ck_dir / "256x256_diffusion_uncond.pt"
        # idempotent + self-repairing: a prior/dangling/wrong-host symlink (e.g. one
        # rsynced from another machine, whose target path does not exist here) reads as
        # not-exists yet makes os.symlink raise FileExistsError. Drop any existing link
        # and (re)point it at the resolved ADM checkpoint.
        if ck_target.is_symlink():
            ck_target.unlink()
        if not ck_target.exists():
            os.symlink(adm, ck_target)
        cwd = os.getcwd()
        os.chdir(_VENDOR)
        try:
            from data.utils import get_score_fn
            return get_score_fn(device=self._dev(), args_path="./libs/eps_ad/args.yml",
                                config_path="./libs/eps_ad/imagenet.yml")
        finally:
            os.chdir(cwd)

    def _load(self):
        if self._model is None:
            self._model = self._build_discriminator()
            self._score_fn = self._build_score_fn()
        return self._model

    def _score_velocity(self, score, image, eps: float = 1e-10):
        import torch
        B, T, C, H, W = score.shape
        pix = torch.zeros_like(score)
        for t in range(1, T - 1):
            pix[:, t] = (image[:, t + 1] - image[:, t - 1]) / 2
        pix[:, 0] = image[:, 1] - image[:, 0]
        pix[:, -1] = image[:, -1] - image[:, -2]
        denom = (score * pix).sum(dim=(2, 3, 4)).view(B, T, 1, 1, 1) + eps
        return score / denom

    def _compute_nsg(self, image):
        """image (B,T,3,224,224) in [0,1] -> NSG velocity (B,T,3,224,224)."""
        import torch
        B, T, C, H, W = image.shape
        curr_t = torch.tensor((self.diffuse_step + 1) / 1000.0, device=image.device)
        flat = image.reshape(B * T, C, H, W)
        score_out = self._score_fn(2.0 * flat - 1.0, curr_t.expand(flat.shape[0]))
        score = torch.split(score_out, score_out.shape[1] // 2, dim=1)[0].view(B, T, C, H, W)
        velocity = self._score_velocity(score, image)
        velocity[:, 0] = velocity[:, 0] * 100.0
        velocity[:, -1] = velocity[:, -1] * 100.0
        return velocity

    def _preprocess(self, clip: Clip):
        import torch
        from PIL import Image
        from vidaudit.detectors._extract import decode_pil
        frames = decode_pil(clip.path, n_frames=self.n_frames, window_sec=2.0)
        out = torch.empty(self.n_frames, 3, self.size, self.size, dtype=torch.float32)
        for t, im in enumerate(frames):
            a = im.resize((self.size, self.size), Image.Resampling.LANCZOS)
            out[t] = torch.from_numpy(np.array(a, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return out.unsqueeze(0).to(self._dev())  # (1, T, 3, 224, 224)

    def features(self, clip: Clip) -> np.ndarray:
        import torch
        model = self._load()
        image = self._preprocess(clip)
        with torch.no_grad():
            nsg = self._compute_nsg(image)
            _, feat = model.net(nsg, out_feature=True)  # (1, 300)
        return feat[0].detach().cpu().numpy().astype(float)
