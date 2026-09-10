from app.tracking import ObjectTracker


def detection(name, box, confidence=0.8):
    return {"class": name, "box": box, "confidence": confidence}


def test_tracker_keeps_an_id_and_reports_dwell_time():
    tracker = ObjectTracker()
    line = {"x1": 0.5, "y1": 0.1, "x2": 0.5, "y2": 0.9}

    first, counts, crossings, dwell = tracker.update(
        "cam", [detection("person", [0.2, 0.2, 0.1, 0.2])], 100.0, line, 10
    )
    second, counts, crossings, dwell = tracker.update(
        "cam", [detection("person", [0.22, 0.2, 0.1, 0.2])], 105.0, line, 10
    )

    assert first[0]["track_id"] == second[0]["track_id"]
    assert counts == {"person": 1}
    assert second[0]["dwell_seconds"] == 5.0
    assert crossings == []
    assert dwell == []


def test_tracker_reports_line_crossing_direction():
    tracker = ObjectTracker()
    line = {"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}
    tracker.update("cam", [detection("car", [0.4, 0.4, 0.1, 0.1])], 100.0, line)

    tracked, _, crossings, _ = tracker.update(
        "cam", [detection("car", [0.55, 0.4, 0.1, 0.1])], 103.0, line
    )

    assert tracked[0]["track_id"] == 1
    assert crossings[0] == {"track_id": 1, "track_uid": tracked[0]["track_uid"], "class": "car", "direction": "A_to_B"}
    assert tracker.counts("cam") == {"A_to_B": 1, "B_to_A": 0, "total": 1}


def test_tracker_emits_one_dwell_event_and_expires_after_missed_ticks():
    tracker = ObjectTracker(max_missed_ticks=1)
    obj = detection("person", [0.2, 0.2, 0.1, 0.2])
    tracker.update("cam", [obj], 100.0, dwell_seconds=10)
    _, _, _, dwell = tracker.update("cam", [detection("person", obj["box"])], 111.0, dwell_seconds=10)
    assert dwell[0]["track_id"] == 1
    assert dwell[0]["class"] == "person"
    assert dwell[0]["dwell_seconds"] == 11.0
    assert len(dwell[0]["track_uid"]) == 32

    tracker.update("cam", [], 114.0)
    tracked, counts, _, _ = tracker.update("cam", [], 117.0)
    assert tracked == []
    assert counts == {}
