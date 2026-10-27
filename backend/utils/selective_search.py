"""
Selective search wrapper. Uses OpenCV's ximgproc implementation of
Uijlings et al. (2013) when available; falls back to the pure-Python
`selectivesearch` package otherwise.
"""
from typing import List
import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = hasattr(cv2, "ximgproc")
except ImportError:
    _HAS_CV2 = False

try:
    import selectivesearch
    _HAS_SS = True
except ImportError:
    _HAS_SS = False


def generate_proposals(image: Image.Image, max_proposals: int = 2000) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))

    if _HAS_CV2:
        return _opencv_ss(arr, max_proposals)
    if _HAS_SS:
        return _python_ss(arr, max_proposals)

    h, w = arr.shape[:2]
    proposals = []
    for size in (64, 128, 256):
        for y in range(0, h - size, size // 2):
            for x in range(0, w - size, size // 2):
                proposals.append([x, y, x + size, y + size])
    return np.array(proposals[:max_proposals], dtype=np.int32)


def _opencv_ss(arr: np.ndarray, max_proposals: int) -> np.ndarray:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(arr)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    if len(rects) > max_proposals:
        areas = rects[:, 2] * rects[:, 3]
        idx = np.argsort(-areas)[:max_proposals]
        rects = rects[idx]
    out = np.zeros((len(rects), 4), dtype=np.int32)
    out[:, 0] = rects[:, 0]
    out[:, 1] = rects[:, 1]
    out[:, 2] = rects[:, 0] + rects[:, 2]
    out[:, 3] = rects[:, 1] + rects[:, 3]
    return out


def _python_ss(arr: np.ndarray, max_proposals: int) -> np.ndarray:
    _, regions = selectivesearch.selective_search(
        arr, scale=500, sigma=0.9, min_size=10
    )
    rects = [r["rect"] for r in regions if r["rect"][2] > 20 and r["rect"][3] > 20]
    if len(rects) > max_proposals:
        areas = [r[2] * r[3] for r in rects]
        order = np.argsort(areas)[::-1][:max_proposals]
        rects = [rects[i] for i in order]
    out = np.zeros((len(rects), 4), dtype=np.int32)
    for i, (x, y, w, h) in enumerate(rects):
        out[i] = [x, y, x + w, y + h]
    return out