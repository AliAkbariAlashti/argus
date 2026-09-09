import time

import numpy as np

from app.camera import CameraFeed


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
