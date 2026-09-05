import threading
import time
import logging
from collections import deque

import cv2

from .config import VIDEOS_DIR, JPEG_QUALITY, MOTION_BUFFER_SECONDS, MOTION_BUFFER_SIZE

log = logging.getLogger("qwenvl.camera")


class CameraFeed:
    """Loops a local video file and keeps the latest decoded frame in memory,
    simulating a live CCTV feed. Real RTSP/webcam sources can later replace
    the file path with a stream URL without changing any consumer code."""

    def __init__(self, cam_id: str, path: str):
        self.id = cam_id
        self.path = path

        self._lock = threading.Lock()
        self._frame = None  # latest raw BGR frame
        self._jpeg = None  # latest encoded JPEG bytes
        # Recent frames spaced ~MOTION_BUFFER_SECONDS apart. A single still
        # frame can't show motion, so questions like "what is the person
        # doing" get a short sequence instead of one snapshot.
        self._history = deque(maxlen=MOTION_BUFFER_SIZE)
        self._last_history_at = 0.0
        self._running = False
        self._thread = None
        # Rolling decode-rate measurement, for the health panel.
        self._frame_times = deque(maxlen=30)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            log.error("Camera %s: could not open video file %s", self.id, self.path)
            self._running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = fps if fps > 1 else 25.0
        delay = 1.0 / fps

        while self._running:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            now = time.monotonic()
            with self._lock:
                self._frame = frame
                if ok:
                    self._jpeg = buf.tobytes()
                if now - self._last_history_at >= MOTION_BUFFER_SECONDS:
                    self._history.append(frame)
                    self._last_history_at = now
                self._frame_times.append(now)

            time.sleep(delay)

        cap.release()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_frame(self):
        """Returns the latest frame as a BGR numpy array, or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_fps(self) -> float:
        """Measured decode rate over the recent window, for the health panel."""
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return round((len(times) - 1) / span, 1) if span > 0 else 0.0

    def get_frame_sequence(self, count: int = MOTION_BUFFER_SIZE):
        """Returns up to `count` recent frames (oldest first, ending with the
        current one) so the model can reason about movement rather than a
        single instant. Falls back to just the latest frame early on, before
        the history buffer has filled."""
        with self._lock:
            frames = [f.copy() for f in self._history][-count:]
            latest = None if self._frame is None else self._frame.copy()

        if latest is None:
            return frames
        # The buffer lags the live frame by up to MOTION_BUFFER_SECONDS;
        # always end the sequence on what the camera shows right now.
        return frames[-(count - 1):] + [latest] if count > 1 else [latest]


class CameraRegistry:
    """Holds the live CameraFeed threads, keyed by camera id. Cameras
    themselves are persisted in Postgres (see models.Camera); this registry
    is just the in-memory runtime side (decoded frames) and is kept in sync
    with the DB by add()/remove() calls from the API layer."""

    def __init__(self):
        self._feeds: dict[str, CameraFeed] = {}
        self._lock = threading.Lock()

    def add(self, cam_id: str, source_path: str):
        with self._lock:
            if cam_id in self._feeds:
                return
            path = str(VIDEOS_DIR / source_path)
            feed = CameraFeed(cam_id, path)
            feed.start()
            self._feeds[cam_id] = feed

    def remove(self, cam_id: str):
        with self._lock:
            feed = self._feeds.pop(cam_id, None)
        if feed:
            feed.stop()

    def get(self, cam_id: str) -> CameraFeed | None:
        return self._feeds.get(cam_id)

    def is_online(self, cam_id: str) -> bool:
        feed = self._feeds.get(cam_id)
        return feed is not None and feed.get_jpeg() is not None

    def stop_all(self):
        with self._lock:
            feeds = list(self._feeds.values())
            self._feeds.clear()
        for feed in feeds:
            feed.stop()


registry = CameraRegistry()
