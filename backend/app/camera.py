import threading
import time
import logging

import cv2

from .config import CAMERAS, VIDEOS_DIR, JPEG_QUALITY

log = logging.getLogger("qwenvl.camera")


class CameraFeed:
    """Loops a local video file and keeps the latest decoded frame in memory,
    simulating a live CCTV feed. Real RTSP/webcam sources can later replace
    the file path with a stream URL without changing any consumer code."""

    def __init__(self, cam_id: str, name: str, location: str, path):
        self.id = cam_id
        self.name = name
        self.location = location
        self.path = str(path)

        self._lock = threading.Lock()
        self._frame = None  # latest raw BGR frame
        self._jpeg = None  # latest encoded JPEG bytes
        self._frame_index = 0
        self._fps = 25.0
        self._running = False
        self._thread = None

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
        self._fps = fps if fps > 1 else 25.0
        delay = 1.0 / self._fps

        while self._running:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            with self._lock:
                self._frame = frame
                if ok:
                    self._jpeg = buf.tobytes()
                self._frame_index += 1

            time.sleep(delay)

        cap.release()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_frame(self):
        """Returns the latest frame as a BGR numpy array, or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class CameraRegistry:
    def __init__(self):
        self._cams: dict[str, CameraFeed] = {}
        for cam in CAMERAS:
            path = VIDEOS_DIR / cam["file"]
            self._cams[cam["id"]] = CameraFeed(
                cam["id"], cam["name"], cam["location"], path
            )

    def start_all(self):
        for cam in self._cams.values():
            cam.start()

    def stop_all(self):
        for cam in self._cams.values():
            cam.stop()

    def get(self, cam_id: str) -> CameraFeed | None:
        return self._cams.get(cam_id)

    def list(self):
        return list(self._cams.values())


registry = CameraRegistry()
