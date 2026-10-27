"""Uniform interface shared by all three detectors."""
from dataclasses import dataclass, field
from typing import List, Dict
import time
import torch


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: List[float]


@dataclass
class DetectionResult:
    model_name: str
    inference_ms: float = 0.0
    detections: List[Detection] = field(default_factory=list)
    proposal_count: int = 0
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "inference_ms": round(self.inference_ms, 2),
            "proposal_count": self.proposal_count,
            "detection_count": len(self.detections),
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "box": [round(v, 2) for v in d.box],
                }
                for d in self.detections
            ],
            "error": self.error,
        }


class BaseDetector:
    name = "base"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = False

    def load(self) -> None:
        raise NotImplementedError

    def detect(self, image_tensor: torch.Tensor, **kwargs) -> DetectionResult:
        raise NotImplementedError

    def _timed_detect(self, fn, *args, **kwargs) -> DetectionResult:
        torch.set_num_threads(1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result