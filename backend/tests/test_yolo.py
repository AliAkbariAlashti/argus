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


def test_detect_passes_xywh_rectangles_to_nms(monkeypatch):
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, :4, 0] = [300, 300, 100, 120]
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = [520, 300, 80, 100]
    output[0, 5, 1] = 0.8

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "images"})()]

        def run(self, _outputs, _inputs):
            return [output]

    captured = {}

    def fake_nms(boxes, scores, score_threshold, iou_threshold):
        captured["boxes"] = boxes
        return np.array([[0], [1]])

    monkeypatch.setattr(yolo, "_session", FakeSession())
    monkeypatch.setattr(yolo.cv2.dnn, "NMSBoxes", fake_nms)

    results = yolo.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert captured["boxes"] == [[250.0, 240.0, 100.0, 120.0], [480.0, 250.0, 80.0, 100.0]]
    assert [row[0] for row in results] == ["person", "bicycle"]
