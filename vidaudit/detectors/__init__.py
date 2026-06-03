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
