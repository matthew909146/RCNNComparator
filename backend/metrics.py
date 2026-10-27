"""mAP@0.5 computation against PASCAL VOC ground truth."""
from typing import Dict, List
import numpy as np
from .config import VOC_CLASSES


def compute_map_at_05(predictions: List[Dict], ground_truths: List[Dict],
                      iou_threshold: float = 0.5) -> float:
    aps = []
    for class_id in range(1, len(VOC_CLASSES)):
        ap = _average_precision_for_class(
            predictions, ground_truths, class_id, iou_threshold
        )
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def _average_precision_for_class(predictions, ground_truths, class_id, iou_thr):
    preds = [p for p in predictions if p["class_id"] == class_id]
    if not preds:
        return None
    preds.sort(key=lambda p: p["confidence"], reverse=True)

    gt_by_image = {}
    for g in ground_truths:
        if g["class_id"] != class_id:
            continue
        gt_by_image.setdefault(g["image_id"], []).append(
            {"box": np.array(g["box"], dtype=np.float32), "matched": False}
        )

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, p in enumerate(preds):
        gts = gt_by_image.get(p["image_id"], [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if g["matched"]:
                continue
            iou = _iou(np.array(p["box"], dtype=np.float32), g["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1
            gts[best_j]["matched"] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, sum(len(v) for v in gt_by_image.values()))
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
        ap += p / 11.0
    return float(ap)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0