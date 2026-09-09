from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app import monitor as module


def test_stationary_pending_detection_gets_confirmation(monkeypatch):
    monitor = module.EventMonitor()
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    feed = Mock()
    feed.get_frame.return_value = frame
    monkeypatch.setattr(module.registry, "get", lambda _: feed)
    monkeypatch.setattr(module, "vlm", SimpleNamespace(interactive=SimpleNamespace(is_set=lambda: False), monitor_enabled=True))
    monkeypatch.setattr(module, "MONITOR_MIN_VLM_INTERVAL_SECONDS", 0)
    monitor._last_gray["cam"] = module._downscale_gray(frame)
    monitor._pending["cam"] = {"person": True}
    monitor._confirmed["cam"] = {"person": False}
    monitor._classify = Mock()
    db = Mock()
    db.query.return_value.filter.return_value.filter.return_value.all.return_value = []
    monitor._check_camera(db, SimpleNamespace(id="cam"))
    monitor._classify.assert_called_once()


def test_string_false_is_not_accepted_as_true(monkeypatch):
    monitor = module.EventMonitor()
    monkeypatch.setattr(module, "vlm", SimpleNamespace(ask=lambda *args, **kwargs: '{"person_present":"false","vehicle_present":false,"weapon_visible":false}'))
    monkeypatch.setattr(module, "build_event_prompt", lambda *args: "check")
    monitor._log_event = Mock()
    monitor._classify(Mock(), SimpleNamespace(id="cam", name="Cam"), np.zeros((90, 160, 3), dtype=np.uint8), [])
    assert monitor._pending == {}
    monitor._log_event.assert_not_called()


def test_new_rule_is_confirmed_only_after_two_readings(monkeypatch):
    monitor = module.EventMonitor()
    monkeypatch.setattr(module, "vlm", SimpleNamespace(ask=lambda *args, **kwargs: '{"person_present":false,"vehicle_present":false,"weapon_visible":false,"custom_matches":["backpack"]}'))
    monkeypatch.setattr(module, "build_event_prompt", lambda *args: "check")
    monitor._pending["cam"] = {"person": False, "vehicle": False, "weapon": False}
    monitor._confirmed["cam"] = dict(monitor._pending["cam"])
    monitor._log_event = Mock()
    camera = SimpleNamespace(id="cam", name="Cam")
    rule = SimpleNamespace(id="rule1", target="backpack")
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    monitor._classify(Mock(), camera, frame, [rule])
    monitor._log_event.assert_not_called()
    monitor._classify(Mock(), camera, frame, [rule])
    monitor._log_event.assert_called_once()
    assert monitor._confirmed["cam"]["rule:rule1"] is True
