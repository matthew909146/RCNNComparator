"""
Faster R-CNN via torchvision's maintained implementation.
"""
from typing import Optional
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..config import (
    VOC_CLASSES, NUM_CLASSES, FASTER_RCNN_PATH,
    NMS_IOU_THRESHOLD, CONFIDENCE_THRESHOLD,
)
from .base import BaseDetector, Detection, DetectionResult


class FasterRCNNDetector(BaseDetector):
    name = "Faster R-CNN"

    def __init__(self, device: str = "cpu", weights_path: Optional[Path] = None,
                 use_coco_weights: bool = False):
        super().__init__(device)
        self.weights_path = weights_path or FASTER_RCNN_PATH
        self.use_coco_weights = use_coco_weights
        self._coco_classes = None

    def load(self) -> None:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
        )
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        if self.weights_path.exists() and not self.use_coco_weights:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
            state = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state)
            self._classes = VOC_CLASSES
        elif self.use_coco_weights:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights as W,
            )
            self._classes = W.DEFAULT.meta["categories"]
        else:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
            self._classes = VOC_CLASSES
            print(f"[Faster R-CNN] WARNING: no VOC checkpoint at "
                  f"{self.weights_path}.")

        model.to(self.device).eval()
        self.model = model
        self._loaded = True

    def detect(self, image, original_pil=None, **kwargs) -> DetectionResult:
        return self._timed_detect(lambda img: self._detect(img, original_pil), image)

    def _detect(self, image: torch.Tensor, original_pil=None) -> DetectionResult:
        result = DetectionResult(model_name=self.name)
        x = image.to(self.device)

        with torch.no_grad():
            output = self.model([x[0]])[0]

        boxes = output["boxes"].cpu().numpy()

        # Rescale boxes back to original image dimensions if provided
        if original_pil is not None:
            orig_w, orig_h = original_pil.size
            _, _, tensor_h, tensor_w = image.shape
            scale_x = orig_w / float(tensor_w)
            scale_y = orig_h / float(tensor_h)
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        result.proposal_count = int(len(boxes))
        for box, lbl, sc in zip(boxes, labels, scores):
            if sc < CONFIDENCE_THRESHOLD:
                continue
            cls_name = (self._classes[lbl] if 0 <= lbl < len(self._classes)
                        else f"class_{lbl}")
            result.detections.append(Detection(
                class_id=int(lbl),
                class_name=cls_name,
                confidence=float(sc),
                box=[float(v) for v in box],
            ))
        result.detections.sort(key=lambda d: d.confidence, reverse=True)
        return result