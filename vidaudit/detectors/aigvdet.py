"""AIGVDet: a two-stream ResNet50 AI-generated video detector (Bai, Lin, Cao; PRCV 2024;
arXiv:2403.16638). A spatial branch runs ResNet50 over RGB frames; an optical branch runs
ResNet50 over an RGB visualization of optical flow between consecutive frames. score() is the
0.5/0.5 fused per-frame p(generated); features() is the 2048-d penultimate (spatial), or 4096-d
with the optical branch, averaged over frames.

Weights: original.pth (spatial) and optical.pth (optical) on the authors' Google Drive
(academic-research-only). load_weights(<dir>) or a single .pth (spatial only); the checkpoints
are ResNet50 state dicts (fc = Linear(2048, 1)), optionally under a "model" key.

Flow is computed with torchvision raft_large (the paper uses princeton-vl raft-things), so the
spatial branch is exact and the optical branch is an approximation.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_IMG = 448
_CLIP_MEAN = (0.485, 0.456, 0.406)
_CLIP_STD = (0.229, 0.224, 0.225)


def _make_colorwheel():
    """Standard Middlebury color wheel (as used by RAFT's flow_viz)."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3), np.float32)
    c = 0
    cw[0:RY, 0] = 255; cw[0:RY, 1] = np.floor(255 * np.arange(RY) / RY); c += RY
    cw[c:c+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG); cw[c:c+YG, 1] = 255; c += YG
    cw[c:c+GC, 1] = 255; cw[c:c+GC, 2] = np.floor(255 * np.arange(GC) / GC); c += GC
    cw[c:c+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB); cw[c:c+CB, 2] = 255; c += CB
    cw[c:c+BM, 2] = 255; cw[c:c+BM, 0] = np.floor(255 * np.arange(BM) / BM); c += BM
    cw[c:c+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR); cw[c:c+MR, 0] = 255
    return cw


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Colorize an [H, W, 2] flow field to an [H, W, 3] uint8 image (Middlebury wheel)."""
    cw = _make_colorwheel()
    ncols = cw.shape[0]
    u, v = flow[..., 0], flow[..., 1]
    rad = np.sqrt(u ** 2 + v ** 2)
    rad_max = rad.max()
    eps = 1e-5
    u = u / (rad_max + eps); v = v / (rad_max + eps)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0
    img = np.zeros((*u.shape, 3), np.uint8)
    for i in range(3):
        col0 = cw[k0, i] / 255.0
        col1 = cw[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        img[..., i] = np.floor(255 * col)
    return img


@register("aigvdet")
class AIGVDet(Detector):
    spec = DetectorSpec(
        name="AIGVDet", backbone="two-stream ResNet50 (RGB + RAFT flow)", family="fusion",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="authors' Google Drive (original.pth + optical.pth); academic-only, not mirrored",
        license="academic research only (non-commercial); see the AIGVDet repo",
        paper="Bai, Lin, Cao; PRCV 2024 (arXiv:2403.16638)",
        notes="two-stream ResNet50; spatial branch exact, optical branch uses torchvision "
              "RAFT + Middlebury flow-viz (approximates the paper's raft-things). "
              "load_weights(dir-with-original.pth+optical.pth) or a single .pth (RGB only).",
    )

    def __init__(self, n_frames: int = 16, size: int = _IMG, device=None):
        self.n_frames = n_frames
        self.size = size
        self._device = device
        self._rgb = None        # spatial ResNet50
        self._flow = None        # optical ResNet50 (None -> RGB-only mode)
        self._raft = None

    # ---- model construction --------------------------------------------------
    def _dev(self):
        import torch
        if self._device is None:
            self._device = ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    @staticmethod
    def _resnet50_head():
        import torchvision
        m = torchvision.models.resnet50(weights=None, num_classes=1)  # fc = Linear(2048, 1)
        return m.eval()

    @staticmethod
    def _embed(model, x):
        """torchvision ResNet50 forward, returning (penultimate 2048-d, logit)."""
        import torch
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x); feat = torch.flatten(x, 1)
        return feat, model.fc(feat)

    def _ensure(self):
        """Build the architecture (random weights if load_weights was not called)."""
        if self._rgb is None:
            self._rgb = self._resnet50_head().to(self._dev())
        return self._rgb

    def _ensure_raft(self):
        if self._raft is None:
            import torchvision
            self._raft = torchvision.models.optical_flow.raft_large(weights=None).eval().to(self._dev())
        return self._raft

    def load_weights(self, path: Optional[str] = None) -> None:
        """Load the spatial (+ optional optical) ResNet50 checkpoints.

        `path` is a directory containing `original.pth` (and optionally `optical.pth`),
        or a single `.pth` file (loaded as the spatial branch; optical disabled)."""
        import torch
        if not path:
            raise ValueError("AIGVDet has no auto-download weights; pass the Drive "
                             "checkpoints: load_weights('<dir with original.pth[, optical.pth]>').")

        def _load(model, ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
            model.load_state_dict(sd)
            return model.eval().to(self._dev())

        if os.path.isdir(path):
            rgb = os.path.join(path, "original.pth")
            opt = os.path.join(path, "optical.pth")
            if not os.path.exists(rgb):
                rgb = next((os.path.join(path, f) for f in sorted(os.listdir(path))
                            if "original" in f and f.endswith(".pth")), None)
            if rgb is None:
                raise FileNotFoundError(f"no original*.pth in {path!r}")
            self._rgb = _load(self._resnet50_head(), rgb)
            if os.path.exists(opt):
                self._flow = _load(self._resnet50_head(), opt)
        else:
            self._rgb = _load(self._resnet50_head(), path)   # single file -> spatial only

    # ---- preprocessing -------------------------------------------------------
    def _to_batch(self, pil_frames):
        """CenterCrop(448) + ToTensor + ImageNet-normalize -> [N, 3, 448, 448]."""
        import torch
        from torchvision import transforms
        tf = transforms.Compose([transforms.CenterCrop((self.size, self.size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(_CLIP_MEAN, _CLIP_STD)])
        return torch.stack([tf(im.convert("RGB")) for im in pil_frames]).to(self._dev())

    def _flow_frames(self, pil_frames):
        """RAFT flow between consecutive frames -> list of colorized PIL RGB images."""
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        raft = self._ensure_raft()
        imgs = [to_tensor(im.convert("RGB")) * 2 - 1 for im in pil_frames]  # RAFT wants [-1, 1]
        # pad H, W to multiples of 8
        _, H, W = imgs[0].shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        out = []
        for a, b in zip(imgs[:-1], imgs[1:]):
            x = torch.nn.functional.pad(a.unsqueeze(0), (0, pw, 0, ph)).to(self._dev())
            y = torch.nn.functional.pad(b.unsqueeze(0), (0, pw, 0, ph)).to(self._dev())
            with torch.no_grad():
                flow = raft(x, y)[-1][0, :, :H, :W].permute(1, 2, 0).cpu().numpy()  # [H,W,2]
            out.append(Image.fromarray(_flow_to_rgb(flow)))
        return out

    def _frames(self, clip: Clip):
        from vidaudit.detectors._extract import decode_pil
        return decode_pil(clip.path, n_frames=self.n_frames)

    # ---- evidence ------------------------------------------------------------
    def _branch(self, model, batch):
        """Return (mean p(generated), mean 2048-d penultimate) over the batch."""
        import torch
        with torch.no_grad():
            feat, logit = self._embed(model, batch)
            prob = torch.sigmoid(logit).mean().item()
        return prob, feat.mean(dim=0).cpu().numpy()

    def score(self, clip: Clip) -> float:
        """Fused p(generated): 0.5 * spatial + 0.5 * optical (spatial-only if no optical ckpt)."""
        frames = self._frames(clip)
        p_rgb, _ = self._branch(self._ensure(), self._to_batch(frames))
        if self._flow is None:
            return float(p_rgb)
        p_flow, _ = self._branch(self._flow, self._to_batch(self._flow_frames(frames)))
        return float(0.5 * p_rgb + 0.5 * p_flow)

    def features(self, clip: Clip) -> np.ndarray:
        """Frame-averaged penultimate: 2048-d (spatial) or 4096-d (spatial + optical)."""
        frames = self._frames(clip)
        _, f_rgb = self._branch(self._ensure(), self._to_batch(frames))
        if self._flow is None:
            return f_rgb
        _, f_flow = self._branch(self._flow, self._to_batch(self._flow_frames(frames)))
        return np.concatenate([f_rgb, f_flow])
