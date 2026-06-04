"""Detector plugin API + zoo.

Importing this package exposes the plugin contract (`Detector`, `DetectorSpec`,
`Clip`) and the registry (`register`, `get`, `all_detectors`). Each detector
wrapper module registers itself on import; uncomment the imports below as each
wrapper is implemented AND verified to reproduce its paper-audit numbers.
"""
from .base import Clip, Detector, DetectorSpec  # noqa: F401
from .registry import all_detectors, get, register  # noqa: F401

# Verified wrappers (enable as each one passes its reproduction check):
from . import temporalspec    # noqa: F401,E402
from . import d3              # noqa: F401,E402
from . import restrav         # noqa: F401,E402
from . import waverep         # noqa: F401,E402
from . import nsgvd           # noqa: F401,E402
from . import fvmd            # noqa: F401,E402
from . import raft            # noqa: F401,E402
from . import clip            # noqa: F401,E402
from . import probe           # noqa: F401,E402  (trainable readout; validates the trainer)
from . import aigvdet         # noqa: F401,E402  (two-stream ResNet50; arXiv:2403.16638)
from . import skyra           # noqa: F401,E402  (Qwen2.5-VL-7B MLLM; arXiv:2512.15693)
from . import videoveritas    # noqa: F401,E402  (Qwen3-VL-8B MLLM, ModelScope; arXiv:2602.08828)
from . import ivy             # noqa: F401,E402  (Qwen2.5-VL-3B MLLM, IVY-FAKE; arXiv:2506.00979)
from . import stall           # noqa: F401,E402  (training-free DINOv3 likelihood; arXiv:2603.15026)
from . import l3de            # noqa: F401,E402  (DINOv2-G + RAFT + UniDepth 3D-CNN; arXiv:2406.19568)
