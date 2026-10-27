"""Image loading and normalisation shared by all three detectors."""
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return img


def to_tensor_normalised(
    image: Image.Image, size: int, mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    original_size = image.size
    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    resized_size = (new_w, new_h)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size, resized_size


def to_variable_size_tensor(image: Image.Image, shorter_side: int = 600):
    w, h = image.size
    scale = shorter_side / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, (w, h), (new_w, new_h)