"""YOLOv8n object detection via ONNX Runtime. CPU-only, no torch.

The .onnx file is committed to this repo (see backend/models/) rather than
downloaded at runtime, so nothing here fetches a model from the network —
consistent with the rest of CPU tools.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("argus.yolo")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
INPUT_SIZE = 640

# COCO's 80 classes, in the fixed order YOLOv8 was trained on — index order
# matters, this is not alphabetical.
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Suggested default severity per class when creating a CPU alert rule — a
# starting point the operator can always override, not a fixed judgment.
# COCO's 80 classes have no gun/rifle class at all; "weapon-adjacent" here
# only covers what YOLOv8n can actually recognize.
_WARNING_CLASSES = {"knife", "scissors", "baseball bat"}
DEFAULT_SEVERITY = {cls: ("warning" if cls in _WARNING_CLASSES else "info") for cls in CLASSES}

_session = None
_load_error = None


def _load():
    global _session, _load_error
    if _session is not None or _load_error is not None:
        return
    try:
        import onnxruntime as ort

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        log.warning("Object detection unavailable: %s", exc)


def available():
    _load()
    return _session is not None


def _letterbox(frame):
    """Resize+pad to a square INPUT_SIZE canvas, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def detect(frame, conf_threshold=0.4, iou_threshold=0.45):
    """Runs YOLOv8n on a BGR frame. Returns [(class_name, confidence, [x, y, w, h])]
    with box coordinates as fractions of the original frame (0-1)."""
    _load()
    if _session is None:
        return []

    height, width = frame.shape[:2]
    canvas, scale, pad_left, pad_top = _letterbox(frame)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]  # HWC -> NCHW

    input_name = _session.get_inputs()[0].name
    output = _session.run(None, {input_name: blob})[0]  # [1, 84, 8400]
    predictions = output[0].T  # [8400, 84]

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    boxes_xywh = predictions[keep, :4]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # Center xywh (in letterboxed-canvas pixels) -> corner xyxy for NMS.
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_threshold, iou_threshold
    )
    if len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    results = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        # Undo letterbox padding/scale back to the original frame, then to fractions.
        x1, x2 = (x1 - pad_left) / scale, (x2 - pad_left) / scale
        y1, y2 = (y1 - pad_top) / scale, (y2 - pad_top) / scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[i])
        if class_id >= len(CLASSES):
            continue
        results.append((
            CLASSES[class_id],
            round(float(confidences[i]), 3),
            [round(float(x1 / width), 4), round(float(y1 / height), 4),
             round(float((x2 - x1) / width), 4), round(float((y2 - y1) / height), 4)],
        ))
    return results
