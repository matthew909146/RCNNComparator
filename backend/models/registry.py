"""Instantiate and hold all three detectors for the lifetime of the process."""
from ..config import DEVICE
from .rcnn import RCNNDetector
from .fast_rcnn import FastRCNNDetector
from .faster_rcnn import FasterRCNNDetector

_DETECTORS = None


def get_detectors():
    global _DETECTORS
    if _DETECTORS is None:
        _DETECTORS = [
            RCNNDetector(device=DEVICE),
            FastRCNNDetector(device=DEVICE),
            FasterRCNNDetector(device=DEVICE, use_coco_weights=True),
        ]
        for d in _DETECTORS:
            d.load()
    return _DETECTORS