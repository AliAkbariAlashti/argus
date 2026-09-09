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
from .models import AlertRule, Camera, Event
from .runtime import vlm

log = logging.getLogger("qwenvl.monitor")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# category/severity for each built-in flag flipping true, and the label for
# it flipping back false. Weapon clearing is deliberately not logged as its
# own row — a disappearing threat isn't the interesting half of that story,
# and it would double the critical-severity row count for every alert.
_BUILTIN_LABELS = {
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
    GPU with live chat, so polling every camera on a fixed schedule isn't an
    option); a structured VLM answer decides *what* happened, including any
    operator-defined custom alert rules folded into the same call. A reading
    only becomes a logged event once it's held for two consecutive checks —
    a single flaky VLM answer shouldn't produce an entered/left pair a
    moment later — and only on an actual state change, so the log reads as
    a timeline of things that happened, not a snapshot every few seconds."""

    def __init__(self):
        self._thread = None
        self._running = False
        self._last_gray: dict[str, np.ndarray] = {}
        self._last_checked: dict[str, float] = {}
        # cam_id -> {flag_key: bool}, the most recent raw VLM reading, not
        # yet confirmed.
        self._pending: dict[str, dict[str, bool]] = {}
        # cam_id -> {flag_key: bool}, the confirmed/logged state.
        self._confirmed: dict[str, dict[str, bool]] = {}
        self._online: dict[str, bool] = {}

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
        db = SessionLocal()
        try:
            cameras = db.query(Camera).all()
            # Offline detection needs no model, so it runs regardless of
            # whether the VLM has finished loading yet.
            for cam in cameras:
                self._check_online(db, cam)
            if vlm.ready:
                for cam in cameras:
                    self._check_camera(db, cam)
        finally:
            db.close()

    def _check_online(self, db, cam: Camera):
        online = registry.is_online(cam.id)
        prev = self._online.get(cam.id)
        self._online[cam.id] = online
        if prev is None or prev == online:
            return
        if online:
            self._log_event(db, cam, "info", "Camera back online", "", None)
        else:
            self._log_event(db, cam, "warning", "Camera went offline", "", None)

    def _check_camera(self, db, cam: Camera):
        if vlm.interactive.is_set() or not vlm.monitor_enabled:
            return
        feed = registry.get(cam.id)
        if feed is None:
            return
        frame = feed.get_frame()
        if frame is None:
            return

        gray = _downscale_gray(frame)
        prev_gray = self._last_gray.get(cam.id)
        self._last_gray[cam.id] = gray
        if prev_gray is None:
            return  # first sighting of this camera - nothing to diff against yet

        pending = self._pending.get(cam.id, {})
        confirmed = self._confirmed.get(cam.id, {})
        needs_confirmation = any(value != confirmed.get(key, False) for key, value in pending.items())
        if not needs_confirmation and _motion_score(prev_gray, gray) < MONITOR_MOTION_THRESHOLD:
            return

        now = time.monotonic()
        if now - self._last_checked.get(cam.id, 0.0) < MONITOR_MIN_VLM_INTERVAL_SECONDS:
            return
        self._last_checked[cam.id] = now

        rules = (
            db.query(AlertRule)
            .filter(AlertRule.enabled.is_(True), AlertRule.source == "vlm")
            .filter((AlertRule.camera_id == cam.id) | (AlertRule.camera_id.is_(None)))
            .all()
        )
        self._classify(db, cam, frame, rules)

    def _classify(self, db, cam: Camera, frame, rules: list[AlertRule]):
        img = frame_to_pil(frame)
        watch_for = [r.target for r in rules]
        try:
            raw = vlm.ask(img, build_event_prompt(cam, watch_for), max_new_tokens=200)
        except Exception:  # noqa: BLE001
            log.exception("Event classification failed for camera %s", cam.name)
            return

        parsed = _parse_event_json(raw)
        if not isinstance(parsed, dict) or any(
            not isinstance(parsed.get(key), bool)
            for key in ("person_present", "vehicle_present", "weapon_visible")
        ) or not isinstance(parsed.get("custom_matches", []), list):
            log.warning("Could not parse event JSON for %s: %r", cam.name, raw)
            return

        matches = {
            str(m).strip().lower() for m in (parsed.get("custom_matches") or []) if str(m).strip()
        }
        new_raw = {
            "person": bool(parsed.get("person_present")),
            "vehicle": bool(parsed.get("vehicle_present")),
            "weapon": bool(parsed.get("weapon_visible")),
        }
        for rule in rules:
            new_raw[f"rule:{rule.id}"] = rule.target.strip().lower() in matches

        prev_raw = self._pending.get(cam.id)
        self._pending[cam.id] = new_raw

        if prev_raw is None:
            return  # Each flag needs two actual readings, even without new motion.

        summary = str(parsed.get("summary") or "").strip()
        confirmed = dict(self._confirmed.get(cam.id, {}))
        rules_by_id = {r.id: r for r in rules}
        snapshot = None  # built lazily - only if something is actually logged

        for key, value in new_raw.items():
            if key not in prev_raw or prev_raw[key] != value:
                continue
            previous = confirmed.get(key, False)
            confirmed[key] = value
            if previous == value:
                continue
            label = self._transition_label(key, value, rules_by_id)
            if label is None:
                continue
            severity, category = label
            if snapshot is None:
                snapshot = image_to_data_uri(img)
            self._log_event(db, cam, severity, category, summary, snapshot)

        self._confirmed[cam.id] = {key: value for key, value in confirmed.items() if key in new_raw}

    def _transition_label(self, key: str, value: bool, rules_by_id: dict):
        if key.startswith("rule:"):
            if not value:
                return None  # a custom rule clearing isn't logged either, same reasoning as weapon
            rule = rules_by_id.get(key.split(":", 1)[1])
            target = rule.target if rule else key
            return "warning", f'Alert rule matched: "{target}"'

        transitions = _BUILTIN_LABELS[key]
        return transitions["on"] if value else transitions["off"]

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
