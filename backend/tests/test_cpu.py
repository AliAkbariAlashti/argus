import numpy as np

from app.cpu import CpuSettings, annotate, detect_faces, detect_people, measure


def test_measure_skips_detectors_by_default():
    settings = CpuSettings()
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    result, gray = measure(frame, None, settings)
    assert "face_count" not in result
    assert "people_count" not in result
    assert gray.shape == (180, 320)


def test_measure_runs_detectors_when_requested():
    settings = CpuSettings(face_alerts=True, people_alerts=True)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    result, _ = measure(frame, None, settings, run_detectors=True)
    assert result["face_count"] == 0
    assert result["people_count"] == 0
    assert result["face_boxes"] == []
    assert result["people_boxes"] == []


def test_detect_faces_and_people_run_without_error_on_blank_frame():
    gray = np.zeros((180, 320), dtype=np.uint8)
    bgr = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detect_faces(gray) == []
    assert detect_people(bgr) == []


def test_detectors_degrade_to_empty_when_objdetect_missing(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module, "_OBJDETECT_AVAILABLE", False)
    gray = np.zeros((180, 320), dtype=np.uint8)
    bgr = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detect_faces(gray) == []
    assert detect_people(bgr) == []


def test_annotate_draws_face_and_people_boxes():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    settings = CpuSettings()
    result = {"boxes": [], "face_boxes": [[0.1, 0.1, 0.2, 0.2]], "people_boxes": [[0.5, 0.5, 0.2, 0.3]]}
    drawn = annotate(frame, result, settings)
    assert drawn.shape == frame.shape
    assert drawn.any()
