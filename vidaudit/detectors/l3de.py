"""L3DE (Chang et al., ICCV 2025; arXiv:2406.19568): a 3D-cue real-vs-synthetic video
detector. Three frozen monocular cues are fused by a small 3D-CNN ("Net"):

    appearance : DINOv2 ViT-G/14 patch tokens (1536-d on a 16x16 grid per frame)
    motion     : RAFT optical flow (2 channels) between consecutive frames
    depth      : UniDepth-v2 metric depth (1 channel)

Each stream is a stack of strided Conv3d blocks that collapse to a T x 8 x 8 volume; the
three are concatenated (384 channels), fused by three more Conv3d layers, flattened, and
read out by fc1 (-> 128-d) + fc2 (-> 1, sigmoid). The frame count T = 24 is baked into
fc1's input dimension, so the released `L3DE.pth` only loads into `Net(T=24)`.

Score polarity (verified against the paper): the sigmoid output is p(REAL) (the simulation
gap is G = 1 - S, higher S = more realistic), so `score()` returns p(generated) =
1 - sigmoid. `features()` returns the 128-d fc1 embedding.

Licensing: the L3DE head + DINOv2 are Apache-2.0 and RAFT is BSD-3, but the depth proxy
uses UniDepth-v2, which is CC-BY-NC-4.0 (code AND weights). A faithful pipeline therefore
inherits the NON-COMMERCIAL restriction; the model-zoo entry is tagged accordingly. Weights
are point-to-source: `L3DE.pth` is on the authors' Google Drive (not mirrored) and
UniDepth-v2 auto-downloads from HuggingFace.

Heavy (DINOv2-ViT-G + RAFT + UniDepth-v2 per clip) and untestable without the gated-free but
large backbones, so a real run belongs on the cluster. As with AIGVDet, the flow branch uses
torchvision RAFT (the paper uses raft-things), so the optical stream is an approximation;
the Net architecture, loading, and polarity are exact. The 3D-CNN is unit-tested with random
weights here; reproduction numbers are a cluster follow-up (clone the repo to diff the exact
conv strides + the flow/depth normalization before a published run).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vidaudit.detectors.base import Clip, Detector, DetectorSpec
from vidaudit.detectors.registry import register

_DINO_MEAN = (0.5, 0.5, 0.5)
_DINO_STD = (0.5, 0.5, 0.5)
_FLOW_HW = (288, 512)      # (H, W) the flow + depth streams expect
_DRIVE_ID = "1wBAAsJPcsT_bIKXetDbd23PKjmCUtb5s"


def _net_module(T: int):
    """Build the L3DE 3D-CNN (`Net`) for a given frame count T. Reimplemented from the
    repo's Apache-2.0 architecture so the released L3DE.pth state_dict loads strict=True."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, T: int):
            super().__init__()
            # appearance stream (in = 1536, DINOv2-G patch tokens)
            self.conv1 = nn.Conv3d(1536, 128, 3, stride=(1, 2, 2), padding=1)
            # motion stream (in = 2, optical flow)
            self.conv4 = nn.Conv3d(2, 16, 3, stride=(1, 3, 4), padding=1)
            self.conv5 = nn.Conv3d(16, 32, 3, stride=(1, 3, 4), padding=1)
            self.conv6 = nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1)
            self.conv7 = nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1)
            # depth stream (in = 1, metric depth)
            self.conv10 = nn.Conv3d(1, 16, 3, stride=(1, 3, 4), padding=1)
            self.conv11 = nn.Conv3d(16, 32, 3, stride=(1, 3, 4), padding=1)
            self.conv12 = nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1)
            self.conv13 = nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1)
            # fusion (3 x 128 = 384 channels in)
            self.conv3 = nn.Conv3d(384, 512, 3, stride=1, padding=1)
            self.conv8 = nn.Conv3d(512, 256, 3, stride=1, padding=1)
            self.conv9 = nn.Conv3d(256, 512, 3, stride=1, padding=1)
            # head (all streams meet at T x 8 x 8)
            self.fc1 = nn.Linear(512 * T * 8 * 8, 128)
            self.drop = nn.Dropout()
            self.fc2 = nn.Linear(128, 1)

        def forward(self, xs):
            x, y, z = xs
            x = F.relu(self.conv1(x))
            y = F.relu(self.conv4(y)); y = F.relu(self.conv5(y))
            y = F.relu(self.conv6(y)); y = F.relu(self.conv7(y))
            z = F.relu(self.conv10(z)); z = F.relu(self.conv11(z))
            z = F.relu(self.conv12(z)); z = F.relu(self.conv13(z))
            x = torch.cat([x, y, z], dim=1)
            x = F.relu(self.conv3(x)); x = F.relu(self.conv8(x)); x = F.relu(self.conv9(x))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)                 # 128-d feature embedding
            y = self.drop(F.relu(x))
            y = self.fc2(y)
            return x, torch.sigmoid(y)      # (feat, p_real)

    return Net(T)


@register("l3de")
class L3DE(Detector):
    spec = DetectorSpec(
        name="L3DE", backbone="DINOv2-ViT-G + RAFT flow + UniDepth-v2 depth -> 3D-CNN",
        family="fusion", published_weights=True, trainable=False, needs_gpu=True,
        weights_url="L3DE.pth on Google Drive (id " + _DRIVE_ID + "; not mirrored); "
                    "UniDepth-v2 auto-downloads from HuggingFace (lpiccinelli/unidepth-v2-vits14)",
        license="CC-BY-NC-4.0 (non-commercial), driven by the UniDepth-v2 depth proxy. "
                "L3DE head + DINOv2 are Apache-2.0, RAFT is BSD-3.",
        paper="Chang et al., ICCV 2025 (arXiv:2406.19568)",
        notes="3-cue 3D-CNN; T=24 baked into fc1 (load into Net(24)). score() = 1 - sigmoid "
              "(sigmoid is p(real)); features() = 128-d fc1 embedding. Heavy (DINOv2-G + RAFT "
              "+ UniDepth-v2) + non-commercial -> cluster. Flow uses torchvision RAFT "
              "(approximates the paper's raft-things). load_weights(<L3DE.pth>).",
    )
    T = 24

    def __init__(self, device=None):
        self._device = device
        self._net = None
        self._dino = None
        self._raft = None
        self._unidepth = None

    def _dev(self):
        import torch
        if self._device is None:
            self._device = ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    # ---- weights -------------------------------------------------------------
    def load_weights(self, path: Optional[str] = None) -> None:
        """Load the released L3DE.pth (a bare state_dict) into Net(T=24), strict=True.

        Point-to-source: fetch L3DE.pth once from the authors' Google Drive (the file id is
        in `weights_url`) and pass the local path; VidAudit does not mirror it."""
        import torch
        if not path:
            raise ValueError("L3DE has no auto-download weights; fetch L3DE.pth from the "
                             "authors' Google Drive (id " + _DRIVE_ID + ") and pass it: "
                             "load_weights('<L3DE.pth>').")
        net = _net_module(self.T)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and not any(k.startswith("conv") for k in sd):
            sd = sd["model"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        self._net = net.eval().to(self._dev())

    def _ensure_net(self):
        if self._net is None:
            self._net = _net_module(self.T).eval().to(self._dev())  # random weights (smoke only)
        return self._net

    # ---- frozen backbones (lazy) --------------------------------------------
    def _ensure_dino(self):
        if self._dino is None:
            import torch
            self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").eval().to(self._dev())
        return self._dino

    def _ensure_raft(self):
        if self._raft is None:
            import torchvision
            self._raft = torchvision.models.optical_flow.raft_large(weights="DEFAULT").eval().to(self._dev())
        return self._raft

    def _ensure_unidepth(self):
        if self._unidepth is None:
            try:
                from unidepth.models import UniDepthV2
            except ImportError as e:
                raise RuntimeError("L3DE depth proxy needs UniDepth-v2 (CC-BY-NC-4.0, "
                                   "non-commercial); `pip install unidepth` and accept its "
                                   "license.") from e
            self._unidepth = UniDepthV2.from_pretrained(
                "lpiccinelli/unidepth-v2-vits14").eval().to(self._dev())
        return self._unidepth

    # ---- per-cue feature extraction -----------------------------------------
    def _frames(self, clip: Clip):
        from vidaudit.detectors._extract import decode_pil
        return [im.convert("RGB") for im in decode_pil(clip.path, n_frames=self.T + 1)]

    def _appearance(self, frames):
        """DINOv2-G patch tokens -> [1, 1536, T, 16, 16]."""
        import torch
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize(_DINO_MEAN, _DINO_STD)])
        dino = self._ensure_dino()
        grids = []
        for im in frames[:self.T]:
            x = tf(im).unsqueeze(0).to(self._dev())
            with torch.no_grad():
                tok = dino.forward_features(x)["x_norm_patchtokens"][0]   # [256, 1536]
            grids.append(tok.reshape(16, 16, 1536))
        vol = torch.stack(grids, dim=0).permute(3, 0, 1, 2).unsqueeze(0)   # [1,1536,T,16,16]
        return vol.float().to(self._dev())

    def _motion(self, frames):
        """RAFT flow between consecutive frames -> [1, 2, T, 288, 512] (approximate)."""
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor
        raft = self._ensure_raft()
        H, W = _FLOW_HW
        flows = []
        for a, b in zip(frames[:self.T], frames[1:self.T + 1]):
            xa = (to_tensor(a) * 2 - 1).unsqueeze(0)
            xb = (to_tensor(b) * 2 - 1).unsqueeze(0)
            _, _, h, w = xa.shape
            ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
            xa = F.pad(xa, (0, pw, 0, ph)).to(self._dev())
            xb = F.pad(xb, (0, pw, 0, ph)).to(self._dev())
            with torch.no_grad():
                fl = raft(xa, xb)[-1][0, :, :h, :w]                       # [2, h, w]
            fl = F.interpolate(fl.unsqueeze(0), size=(H, W), mode="nearest")[0]
            flows.append(fl)
        vol = torch.stack(flows, dim=1).unsqueeze(0)                      # [1,2,T,H,W]
        return vol.float().to(self._dev())

    def _depth(self, frames):
        """UniDepth-v2 metric depth -> [1, 1, T, 288, 512], clipped + min-max normalized."""
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor
        model = self._ensure_unidepth()
        H, W = _FLOW_HW
        maps = []
        for im in frames[:self.T]:
            rgb = (to_tensor(im) * 255.0).to(self._dev())
            with torch.no_grad():
                depth = model.infer(rgb)["depth"].squeeze()              # [h, w]
            depth = F.interpolate(depth[None, None], size=(H, W), mode="nearest")[0, 0]
            depth = torch.clamp(depth, max=10000.0)
            lo, hi = depth.min(), depth.max()
            depth = (depth - lo) / (hi - lo + 1e-8)
            maps.append(depth)
        vol = torch.stack(maps, dim=0).unsqueeze(0).unsqueeze(0)         # [1,1,T,H,W]
        return vol.float().to(self._dev())

    def _forward(self, clip: Clip):
        frames = self._frames(clip)
        net = self._ensure_net()
        xs = [self._appearance(frames), self._motion(frames), self._depth(frames)]
        import torch
        with torch.no_grad():
            feat, p_real = net(xs)
        return feat.squeeze().cpu().numpy(), float(p_real.squeeze().cpu().numpy())

    # ---- evidence ------------------------------------------------------------
    def score(self, clip: Clip) -> float:
        """p(generated) = 1 - L3DE sigmoid (the sigmoid is p(real))."""
        _, p_real = self._forward(clip)
        return float(1.0 - p_real)

    def features(self, clip: Clip) -> np.ndarray:
        """The 128-d fc1 embedding."""
        feat, _ = self._forward(clip)
        return np.asarray(feat, dtype=np.float64).ravel()
