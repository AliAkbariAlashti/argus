from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.cpu import CpuMonitor, CpuSettings, annotate, detect_faces, detect_people, events_csv, measure


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


def _mock_db(rule=None):
    """A rule lookup uses .query().filter().filter().first(); give every
    Event a real auto-incrementing id via db.flush(), since the dedup logic
    reads event.id right after adding it."""
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = rule
    next_id = iter(range(1, 100000))
    def flush():
        for call in db.add.call_args_list:
            obj = call.args[0]
            if getattr(obj, "id", None) is None:
                obj.id = next(next_id)
    db.flush.side_effect = flush
    return db


def test_object_only_detection_is_committed(monkeypatch):
    # Regression: db.add()/db.flush() without a matching db.commit() means a
    # real SQLAlchemy session rolls everything back when it closes — a real
    # bug this suite's Mock-based db didn't catch until this test was added,
    # because Mock() doesn't simulate an uncommitted transaction disappearing.
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("car", 0.8, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    # Only object detection enabled — motion/quality/face/people all off, so
    # the only path that could log anything is the object-detection branch.
    settings = CpuSettings(motion_alerts=False, quality_alerts=False, face_alerts=False,
                           people_alerts=False, object_alerts=True)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    monitor.process(db, camera, frame, settings)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert event_adds, "expected an Event to be added"
    from app.models import Observation
    observation_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], Observation)]
    assert len(observation_adds) == 1
    assert observation_adds[0].kind == "object_detection"
    assert observation_adds[0].object_type == "car"
    assert observation_adds[0].box == [0.1, 0.1, 0.2, 0.2]
    assert db.commit.called, "Event was added but never committed — it would be silently rolled back"


def test_tracking_crossing_and_dwell_write_structured_observations(monkeypatch):
    from app import cpu as module
    from app.models import Observation

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("person", 0.91, [0.2, 0.2, 0.2, 0.3])])
    monitor = CpuMonitor()

    def tracked(camera_id, detections, now, line=None, dwell_seconds=None):
        detections[0].update({"track_id": 7, "track_uid": "a" * 32, "dwell_seconds": 12.0})
        crossing = {"track_id": 7, "track_uid": "a" * 32, "class": "person", "direction": "A_to_B"}
        dwell = {"track_id": 7, "track_uid": "a" * 32, "class": "person", "dwell_seconds": 12.0}
        return detections, {"person": 1}, [crossing], [dwell]

    monkeypatch.setattr(monitor._tracker, "update", tracked)
    settings = CpuSettings(object_alerts=True, tracking_enabled=True, line_crossing_alerts=True,
                           dwell_alerts=True, dwell_seconds=10, motion_alerts=False)
    db = _mock_db()
    monitor.process(db, SimpleNamespace(id="cam1"), np.zeros((180, 320, 3), dtype=np.uint8), settings)
    observations = [call.args[0] for call in db.add.call_args_list if isinstance(call.args[0], Observation)]
    assert {row.kind for row in observations} == {"object_detection", "line_crossing", "dwell"}
    assert all(row.track_id == "a" * 32 for row in observations)
    crossing = next(row for row in observations if row.kind == "line_crossing")
    assert crossing.direction == "A_to_B"


def test_process_logs_an_event_every_detector_tick_no_gating(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda frame, conf_threshold=0.4: [("car", 0.8, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    # _DETECT_EVERY_N_TICKS is 6 (ticks 0, 6, 12, ... run the detector), so 7
    # ticks (0-6) log an Event on every detector tick — 2 total.
    _run_process_ticks(monitor, camera, frame, settings, 7, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert len(event_adds) == 2
    assert event_adds[0].category == "CPU · Car detected"
    assert event_adds[0].confidence == 80


def test_process_logs_nothing_when_no_detector_enabled(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(face_alerts=False, people_alerts=False, object_alerts=False, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    _run_process_ticks(monitor, camera, frame, settings, 7, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert event_adds == []


def test_object_detection_logs_even_with_no_matching_rule(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("dog", 0.7, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db(rule=None)  # no AlertRule exists for "dog"

    monitor.process(db, camera, frame, settings)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert len(event_adds) == 1
    assert event_adds[0].category == "CPU · Dog detected"
    assert event_adds[0].severity == "info"  # default severity, no gate on logging it


def test_cpu_rule_match_uses_the_rules_own_severity(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("knife", 0.9, [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    rule = SimpleNamespace(target="knife", severity="critical")
    db = _mock_db(rule=rule)

    monitor.process(db, camera, frame, settings)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert event_adds
    assert event_adds[0].severity == "critical"
    assert event_adds[0].category == "CPU · Knife detected"


def test_motion_and_quality_events_keep_their_fixed_severity(monkeypatch):
    from app import cpu as module

    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(motion_alerts=True, quality_alerts=True, cooldown_seconds=5,
                           dark_threshold=150, blur_threshold=1000)  # max thresholds; a black frame is always below both
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    monitor.process(db, camera, frame, settings)  # warm-up tick, no previous frame
    monitor.process(db, camera, frame, settings)
    monitor.process(db, camera, frame, settings)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    severities = {e.category: e.severity for e in event_adds}
    assert severities.get("CPU · Low-light frame") == "warning"
    assert severities.get("CPU · Low detail / possible blur") == "warning"


def test_snapshot_dedup_keeps_highest_confidence_in_a_run(monkeypatch):
    from app import cpu as module

    confidences = iter([0.5, 0.9, 0.6])
    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: [("knife", next(confidences), [0.1, 0.1, 0.2, 0.2])])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    # Only ticks 0, 6, 12 (multiples of _DETECT_EVERY_N_TICKS) actually run
    # the detector — 13 ticks gives exactly 3 detector samples.
    _run_process_ticks(monitor, camera, frame, settings, 13, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert len(event_adds) == 3
    # A Mock DB doesn't apply the UPDATE to event_adds[0] in-memory (a real
    # DB would), so assert on the UPDATE call itself: exactly one, clearing
    # a snapshot, fired when the second (higher-confidence) sample arrived.
    assert event_adds[1].snapshot is not None
    assert event_adds[2].snapshot is None
    update_calls = db.query.return_value.filter.return_value.update.call_args_list
    assert len(update_calls) == 1
    assert update_calls[0].args[0] == {"snapshot": None}


def test_snapshot_dedup_resets_after_a_gap_in_the_run(monkeypatch):
    from app import cpu as module

    detections = iter([
        [("knife", 0.5, [0.1, 0.1, 0.2, 0.2])],
        [],  # gap: knife not seen this detector tick, run ends
        [("knife", 0.4, [0.1, 0.1, 0.2, 0.2])],  # new run: lower confidence than before, still keeps a snapshot
    ])
    monkeypatch.setattr(module.yolo, "detect", lambda *a, **k: next(detections))
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(object_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    _run_process_ticks(monitor, camera, frame, settings, 13, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert len(event_adds) == 2  # the empty detector tick logs nothing
    assert event_adds[0].snapshot is not None
    assert event_adds[1].snapshot is not None  # new run after the gap, not compared to the old one


def test_face_and_people_detection_keep_only_first_snapshot_in_a_run(monkeypatch):
    from app import cpu as module

    monkeypatch.setattr(module, "detect_faces", lambda gray: [[0.1, 0.1, 0.2, 0.2]])
    monitor = CpuMonitor()
    camera = SimpleNamespace(id="cam1")
    settings = CpuSettings(face_alerts=True, motion_alerts=False)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    db = _mock_db()

    _run_process_ticks(monitor, camera, frame, settings, 7, db)

    event_adds = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], module.Event)]
    assert len(event_adds) == 2
    assert event_adds[0].snapshot is not None
    assert event_adds[1].snapshot is None  # no confidence to compare: first-in-run keeps it


def test_events_csv_includes_confidence_column():
    event = SimpleNamespace(to_dict=lambda: {
        "id": 1, "camera_id": "cam1", "created_at": "2026-01-01T00:00:00+00:00",
        "severity": "info", "category": "CPU · Car detected", "confidence": 87, "summary": "test",
    })
    csv_text = events_csv([event])
    header, row = csv_text.strip().split("\r\n")
    assert header == "id,camera_id,created_at,severity,category,confidence,summary"
    assert row == "1,cam1,2026-01-01T00:00:00+00:00,info,CPU · Car detected,87,test"
