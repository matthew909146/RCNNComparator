"""
Fast R-CNN reimplementation (Girshick, 2015).
"""
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights
from torchvision.ops import roi_pool

from ..config import (
    VOC_CLASSES, NUM_CLASSES, VGG_INPUT_SIZE, NMS_IOU_THRESHOLD,
    CONFIDENCE_THRESHOLD, RCNN_PROPOSAL_COUNT, FAST_RCNN_HEAD_PATH,
)
from ..utils.selective_search import generate_proposals
from .base import BaseDetector, Detection, DetectionResult
from .rcnn import _nms


class FastRCNNHead(nn.Module):
    def __init__(self, in_features: int = 512 * 7 * 7,
                 fc_dim: int = 4096, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_features, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.drop = nn.Dropout(0.5)
        self.cls = nn.Linear(fc_dim, num_classes)
        self.bbox = nn.Linear(fc_dim, num_classes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pooled: torch.Tensor):
        x = pooled.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.cls(x), self.bbox(x)


class FastRCNNDetector(BaseDetector):
    name = "Fast R-CNN"

    def __init__(self, device: str = "cpu", head_path: Optional[Path] = None):
        super().__init__(device)
        self.head_path = head_path or FAST_RCNN_HEAD_PATH

    def load(self) -> None:
        vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.backbone = vgg.features[:31].to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = FastRCNNHead().to(self.device)
        if self.head_path.exists():
            state = torch.load(self.head_path, map_location=self.device)
            self.head.load_state_dict(state)
        else:
            print(f"[Fast R-CNN] WARNING: no head checkpoint at {self.head_path}.")
        self.head.eval()
        self._loaded = True

    def detect(self, image, original_pil=None, **kwargs) -> DetectionResult:
        return self._timed_detect(self._detect, image, original_pil)

    def _detect(self, image: torch.Tensor, original_pil) -> DetectionResult:
        result = DetectionResult(model_name=self.name)

        proposals = generate_proposals(original_pil, RCNN_PROPOSAL_COUNT)
        result.proposal_count = len(proposals)
        if len(proposals) == 0:
            return result

        with torch.no_grad():
            feat_map = self.backbone(image.to(self.device))
            _, _, Hf, Wf = feat_map.shape
            _, _, Hi, Wi = image.shape
            scale_x, scale_y = Wf / Wi, Hf / Hi
            rois = torch.tensor(proposals, dtype=torch.float32, device=self.device)
            rois[:, [0, 2]] *= scale_x
            rois[:, [1, 3]] *= scale_y
            batch_idx = torch.zeros(len(rois), 1, device=self.device)
            rois = torch.cat([batch_idx, rois], dim=1)

            pooled = roi_pool(feat_map, rois,
                              output_size=(7, 7), spatial_scale=1.0)
            cls_logits, bbox_deltas = self.head(pooled)
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            deltas = bbox_deltas.cpu().numpy()

        detections: List[Detection] = []
        from .rcnn import RCNNDetector
        decoder = RCNNDetector.__new__(RCNNDetector)
        for c in range(1, NUM_CLASSES):
            scores = probs[:, c]
            keep = scores > CONFIDENCE_THRESHOLD
            if not keep.any():
                continue
            boxes_c = proposals[keep].astype(np.float32)
            scores_c = scores[keep]
            boxes_c = decoder._apply_bbox_regression(
                boxes_c, deltas[keep], np.full(int(keep.sum()), c)
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