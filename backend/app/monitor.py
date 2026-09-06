import json
import logging
import re
import threading
import time

import cv2
import numpy as np

from .camera import registry
from .config import (
    MONITOR_ENABLED,
    MONITOR_MIN_VLM_INTERVAL_SECONDS,
    MONITOR_MOTION_THRESHOLD,
    MONITOR_POLL_SECONDS,
)
from .db import SessionLocal
from .grounding import build_event_prompt
from .imaging import frame_to_pil, image_to_data_uri
from .models import Camera, Event
from .vlm import vlm

log = logging.getLogger("qwenvl.monitor")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# category, severity for each tracked flag flipping true, and the label for
# it flipping back false. Weapon clearing is deliberately not logged as its
# own row — a disappearing threat isn't the interesting half of that story,
# and it would double the critical-severity row count for every alert.
_TRANSITIONS = {
    "weapon": {"on": ("critical", "Possible weapon visible"), "off": None},
    "person": {"on": ("info", "Person entered view"), "off": ("info", "Person left view")},
    "vehicle": {"on": ("info", "Vehicle entered view"), "off": ("info", "Vehicle left view")},
}


def _downscale_gray(frame):
    small = cv2.resize(frame, (160, 90))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _motion_score(a, b) -> float:
    diff = cv2.absdiff(a, b)
    return float(np.count_nonzero(diff > 25)) / diff.size


def _parse_event_json(raw: str):
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class EventMonitor:
    """Background loop, one thread for the whole fleet: cheap per-camera
    motion diffing decides *when* to bother asking the VLM (this shares one
    GPU with live chat, so polling every camera on a fixed schedule isn't
    an option), and a structured yes/no VLM answer decides *what* happened.
    Events are logged only on a state change, so the log reads as a
    timeline of things that happened, not a snapshot every few seconds."""

    def __init__(self):
        self._thread = None
        self._running = False
        self._last_gray: dict[str, np.ndarray] = {}
        self._last_checked: dict[str, float] = {}
        self._state: dict[str, dict[str, bool]] = {}

    def start(self):
        if not MONITOR_ENABLED or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Event monitor started.")

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                log.exception("Monitor tick failed")
            time.sleep(MONITOR_POLL_SECONDS)

    def _tick(self):
        if not vlm.ready:
            return
        db = SessionLocal()
        try:
            for cam in db.query(Camera).all():
                self._check_camera(db, cam)
        finally:
            db.close()

    def _check_camera(self, db, cam: Camera):
        feed = registry.get(cam.id)
        if feed is None:
            return
        frame = feed.get_frame()
        if frame is None:
            return

        gray = _downscale_gray(frame)
        prev = self._last_gray.get(cam.id)
        self._last_gray[cam.id] = gray
        if prev is None:
            return  # first sighting of this camera - nothing to diff against yet

        if _motion_score(prev, gray) < MONITOR_MOTION_THRESHOLD:
            return

        now = time.monotonic()
        if now - self._last_checked.get(cam.id, 0.0) < MONITOR_MIN_VLM_INTERVAL_SECONDS:
            return
        self._last_checked[cam.id] = now

        self._classify(db, cam, frame)

    def _classify(self, db, cam: Camera, frame):
        img = frame_to_pil(frame)
        try:
            raw = vlm.ask(img, build_event_prompt(cam), max_new_tokens=150)
        except Exception:  # noqa: BLE001
            log.exception("Event classification failed for camera %s", cam.name)
            return

        parsed = _parse_event_json(raw)
        if parsed is None:
            log.warning("Could not parse event JSON for %s: %r", cam.name, raw)
            return

        prev_state = self._state.get(cam.id, {"person": False, "vehicle": False, "weapon": False})
        new_state = {
            "person": bool(parsed.get("person_present")),
            "vehicle": bool(parsed.get("vehicle_present")),
            "weapon": bool(parsed.get("weapon_visible")),
        }
        self._state[cam.id] = new_state
        summary = str(parsed.get("summary") or "").strip()
        snapshot = image_to_data_uri(img)

        for key, transitions in _TRANSITIONS.items():
            if new_state[key] and not prev_state[key]:
                severity, category = transitions["on"]
            elif prev_state[key] and not new_state[key] and transitions["off"]:
                severity, category = transitions["off"]
            else:
                continue
            self._log_event(db, cam, severity, category, summary, snapshot)

    def _log_event(self, db, cam, severity, category, summary, snapshot):
        event = Event(
            camera_id=cam.id,
            severity=severity,
            category=category,
            summary=summary or category,
            snapshot=snapshot,
        )
        db.add(event)
        db.commit()
        log.info("Event[%s] %s: %s", severity, cam.name, category)


monitor = EventMonitor()
