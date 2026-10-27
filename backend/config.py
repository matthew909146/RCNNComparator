"""Central configuration for the RCNN Family Comparator backend."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

NMS_IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
RCNN_PROPOSAL_COUNT = 2000
RCNN_BATCH_SIZE = 128
RCNN_INPUT_SIZE = 227
FRCNN_INPUT_SIZE = 600
VGG_INPUT_SIZE = 600
MAX_UPLOAD_MB = 10

VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)

DEVICE = os.getenv("RCNN_DEVICE", "cpu")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'comparator.db'}",
)
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

RCNN_HEAD_PATH = WEIGHTS_DIR / "rcnn_head.pt"
FAST_RCNN_HEAD_PATH = WEIGHTS_DIR / "fast_rcnn_head.pt"
FASTER_RCNN_PATH = WEIGHTS_DIR / "faster_rcnn_voc.pt"