import numpy as np

from app import yolo


def test_detect_degrades_to_empty_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo, "_session", None)
    monkeypatch.setattr(yolo, "_load_error", None)
    monkeypatch.setattr(yolo, "MODEL_PATH", tmp_path / "does-not-exist.onnx")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert yolo.detect(frame) == []
    assert yolo.available() is False


def test_letterbox_preserves_aspect_ratio():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, scale, pad_left, pad_top = yolo._letterbox(frame)
    assert canvas.shape == (yolo.INPUT_SIZE, yolo.INPUT_SIZE, 3)
    assert scale == yolo.INPUT_SIZE / 640
    assert pad_left == 0
    assert pad_top > 0
