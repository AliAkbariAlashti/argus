import time

import numpy as np

from app.camera import CameraFeed, CameraRegistry


def test_stale_and_stopped_feeds_never_supply_evidence():
    feed = CameraFeed("test", "unused")
    feed._running = True
    feed._frame = np.zeros((16, 16, 3), dtype=np.uint8)
    feed._jpeg = b"test-jpeg"
    feed._frame_times.append(time.monotonic() - 10)
    assert not feed.is_online()
    assert feed.get_frame() is None
    assert feed.get_jpeg() is None
    assert feed.get_frame_sequence() == []
    feed._frame_times.append(time.monotonic())
    assert feed.is_online()
    assert feed.get_frame() is not None
    feed._running = False
    assert not feed.is_online()
    assert feed.get_jpeg() is None


def test_registry_keeps_rtsp_url_instead_of_treating_it_as_a_video_file(monkeypatch):
    monkeypatch.setattr(CameraFeed, "start", lambda self: None)
    registry = CameraRegistry()
    url = "rtsp://192.168.1.50:8554/camera"
    registry.add("live", url, "rtsp")
    feed = registry.get("live")
    assert feed.path == url
    assert feed.source_type == "rtsp"


def test_rtsp_feed_retries_when_camera_is_temporarily_unavailable(monkeypatch):
    feed = CameraFeed("live", "rtsp://192.168.1.50:8554/camera", "rtsp")
    attempts = []

    class ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    def open_capture(path):
        attempts.append(path)
        if len(attempts) == 2:
            feed._running = False
        return ClosedCapture()

    monkeypatch.setattr("app.camera.cv2.VideoCapture", open_capture)
    monkeypatch.setattr("app.camera.time.sleep", lambda _: None)
    feed._running = True
    feed._run()
    assert attempts == [feed.path, feed.path]


def test_rtsp_credentials_are_redacted_from_log_source():
    feed = CameraFeed("live", "rtsp://viewer:secret@192.168.1.50:8554/camera?profile=main", "rtsp")
    assert feed._log_source() == "rtsp://192.168.1.50:8554/camera?profile=main"
