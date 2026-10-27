"""
Original R-CNN reimplementation (Girshick et al., 2014).
"""
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import AlexNet_Weights

from ..config import (
    VOC_CLASSES, NUM_CLASSES, RCNN_INPUT_SIZE, RCNN_BATCH_SIZE,
    RCNN_PROPOSAL_COUNT, NMS_IOU_THRESHOLD, CONFIDENCE_THRESHOLD,
    RCNN_HEAD_PATH,
)
from ..preprocessing import IMAGENET_MEAN, IMAGENET_STD
from ..utils.selective_search import generate_proposals
from .base import BaseDetector, Detection, DetectionResult


class RCNNHeads(nn.Module):
    def __init__(self, feature_dim: int = 4096, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cls = nn.Linear(feature_dim, num_classes)
        self.bbox = nn.Linear(feature_dim, num_classes * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.cls.bias)
        nn.init.normal_(self.bbox.weight, std=0.001)
        nn.init.zeros_(self.bbox.bias)

    def forward(self, features: torch.Tensor):
        return self.cls(features), self.bbox(features)


class RCNNDetector(BaseDetector):
    name = "R-CNN"

    def __init__(self, device: str = "cpu", head_path: Optional[Path] = None):
        super().__init__(device)
        self.head_path = head_path or RCNN_HEAD_PATH

    def load(self) -> None:
        self.backbone = torchvision.models.alexnet(
            weights=AlexNet_Weights.IMAGENET1K_V1
        ).to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.heads = RCNNHeads().to(self.device)
        if self.head_path.exists():
            state = torch.load(self.head_path, map_location=self.device)
            self.heads.load_state_dict(state)
        else:
            print(f"[R-CNN] WARNING: no head checkpoint at {self.head_path}.")
        self.heads.eval()
        self._loaded = True

    def _warp_batch_proposals(self, image: torch.Tensor, boxes: np.ndarray) -> torch.Tensor:
        _, _, H, W = image.shape
        patches = []
        for x1, y1, x2, y2 in boxes:
            x1c, y1c = max(0, int(x1)), max(0, int(y1))
            x2c, y2c = min(W, int(x2)), min(H, int(y2))
            if x2c <= x1c or y2c <= y1c:
                patches.append(torch.zeros(3, RCNN_INPUT_SIZE, RCNN_INPUT_SIZE))
                continue
            crop = image[0, :, y1c:y2c, x1c:x2c]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(RCNN_INPUT_SIZE, RCNN_INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )[0]
            patches.append(crop)
        return torch.stack(patches, dim=0)

    def _forward_batch(self, batch_patches: torch.Tensor) -> tuple:
        clf = self.backbone.classifier
        batch = batch_patches.to(self.device)
        x = self.backbone.features(batch)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = clf[0](x); x = clf[1](x); x = clf[2](x)
        x = clf[3](x); x = clf[4](x); x = clf[5](x)
        cls_logits, bbox_deltas = self.heads(x)
        probs = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
        deltas = bbox_deltas.detach().cpu().numpy()
        return probs, deltas

    def _apply_bbox_regression(self, boxes: np.ndarray, deltas: np.ndarray,
                               labels: np.ndarray) -> np.ndarray:
        out = boxes.astype(np.float32).copy()
        for i in range(len(boxes)):
            d = deltas[i, labels[i] * 4: labels[i] * 4 + 4]
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            dx, dy, dw, dh = d
            cx += dx * w; cy += dy * h
            w *= np.exp(dw); h *= np.exp(dh)
            out[i] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        return out

    def detect(self, image, original_pil=None, **kwargs) -> DetectionResult:
        return self._timed_detect(self._detect, image, original_pil)

    def _detect(self, image: torch.Tensor, original_pil) -> DetectionResult:
        result = DetectionResult(model_name=self.name)
        proposals = generate_proposals(original_pil, RCNN_PROPOSAL_COUNT)
        result.proposal_count = len(proposals)
        if len(proposals) == 0:
            return result

        probs_list, deltas_list = [], []
        with torch.no_grad():
            for i in range(0, len(proposals), RCNN_BATCH_SIZE):
                batch_boxes = proposals[i:i + RCNN_BATCH_SIZE]
                batch_patches = self._warp_batch_proposals(image, batch_boxes)
                p_b, d_b = self._forward_batch(batch_patches)
                probs_list.append(p_b)
                deltas_list.append(d_b)
                del batch_patches, p_b, d_b
            probs = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, NUM_CLASSES))
            deltas = np.concatenate(deltas_list, axis=0) if deltas_list else np.empty((0, NUM_CLASSES * 4))

        detections: List[Detection] = []
        for c in range(1, NUM_CLASSES):
            scores = probs[:, c]
            keep = scores > CONFIDENCE_THRESHOLD
            if not keep.any():
                continue
            boxes_c = proposals[keep].astype(np.float32)
            scores_c = scores[keep]
            boxes_c = self._apply_bbox_regression(
                boxes_c, deltas[keep], np.full(keep.sum(), c)
            )
            keep_idx = _nms(boxes_c, scores_c, NMS_IOU_THRESHOLD)
            for j in keep_idx:
                detections.append(Detection(
                    class_id=c, class_name=VOC_CLASSES[c],
                    confidence=float(scores_c[j]),
                    box=[float(v) for v in boxes_c[j]],
                ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        result.detections = detections[:100]
        return result


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep