from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.cpu import CpuMonitor, CpuSettings, annotate, detect_faces, detect_people, measure


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


def test_annotate_draws_object_boxes_with_labels():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    settings = CpuSettings()
    result = {"boxes": [], "objects": [{"class": "person", "confidence": 0.87, "box": [0.2, 0.2, 0.3, 0.4]}]}
    drawn = annotate(frame, result, settings)
    assert drawn.shape == frame.shape
    assert drawn.any()


def test_measure_skips_object_detection_when_model_unavailable(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    settings = CpuSettings(object_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    result, _ = measure(frame, None, settings, run_detectors=True)
    assert "objects" not in result


def test_measure_runs_object_detection_when_enabled(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda frame, conf_threshold=0.4: [("person", 0.9, [0.1, 0.1, 0.2, 0.2])])
    settings = CpuSettings(object_alerts=True)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    result, _ = measure(frame, None, settings, run_detectors=True)
    assert result["objects"] == [{"class": "person", "confidence": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}]


def _run_process_ticks(monitor, camera, frame, settings, count, db):
    for _ in range(count):
        monitor.process(db, camera, frame, settings)


def test_process_logs_a_detection_row_every_detector_tick(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda frame, conf_threshold=0.4: [("car", 0.8, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = []

    # _DETECT_EVERY_N_TICKS is 6 (ticks 0, 6, 12, ... run the detector), so 7
    # ticks (0-6) should log exactly 2 Detection rows: tick 0 and tick 6.
    _run_process_ticks(monitor, camera, frame, settings, 7, db)

    detection_adds = [c for c in db.add.call_args_list if isinstance(c.args[0], module.Detection)]
    assert len(detection_adds) == 2
    assert detection_adds[0].args[0].objects == [{"class": "car", "confidence": 0.8}]


def test_process_does_not_log_detection_row_when_nothing_enabled(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(face_alerts=False, people_alerts=False, object_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = []

    _run_process_ticks(monitor, camera, frame, settings, 7, db)

    detection_adds = [c for c in db.add.call_args_list if isinstance(c.args[0], module.Detection)]
    assert detection_adds == []


def test_prune_detections_throttled_to_once_per_ten_minutes():
    monitor = CpuMonitor()
    db = Mock()
    monitor._prune_detections(db)
    db.query.assert_called_once()
    db.commit.assert_called_once()
    monitor._prune_detections(db)
    db.query.assert_called_once()  # second call within the window is a no-op


def test_cpu_rule_match_uses_the_rules_own_severity(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("knife", 0.9, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False, cooldown_seconds=5)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    rule = SimpleNamespace(target="knife", severity="critical")
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = [rule]

    # Two detector ticks needed before a candidate is confirmed (streak >= 2).
    _run_process_ticks(monitor, camera, frame, settings, 13, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert event_adds, "expected at least one Event to be logged"
    assert event_adds[0].severity == "critical"
    assert event_adds[0].category == "CPU · Knife detected"


def test_motion_and_quality_events_keep_their_fixed_severity(monkeypatch):
    from app import cpu as module

    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(motion_alerts=True, quality_alerts=True, cooldown_seconds=5,
                           dark_threshold=150, blur_threshold=1000)  # max thresholds; a black frame is always below both
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = []

    monitor.process(db, camera, frame, settings)  # warm-up tick, no previous frame
    monitor.process(db, camera, frame, settings)
    monitor.process(db, camera, frame, settings)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    severities = {e.category: e.severity for e in event_adds}
    assert severities.get("CPU · Low-light frame") == "warning"
    assert severities.get("CPU · Low detail / possible blur") == "warning"
