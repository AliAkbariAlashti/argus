"""CPU-only video measurements. No model weights, remote calls, or GPU imports."""
import csv
import io
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from . import yolo
from .camera import registry
from .db import SessionLocal
from .imaging import frame_to_pil, image_to_data_uri
from .models import AlertRule, Camera, CpuConfig, Detection, Event

# How long to keep the dense per-tick detection log before pruning.
DETECTION_RETENTION_HOURS = 24

log = logging.getLogger("argus.cpu")

# Some opencv-python-headless wheels (5.x at time of writing) ship without the
# objdetect module: no CascadeClassifier/HOGDescriptor. Degrade face/people
# detection to "unavailable" instead of crashing motion/brightness/blur, which
# don't need objdetect at all.
try:
    _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    _OBJDETECT_AVAILABLE = True
except AttributeError:
    _FACE_CASCADE = _HOG = None
    _OBJDETECT_AVAILABLE = False
    log.warning("This OpenCV build has no objdetect module (CascadeClassifier/HOGDescriptor missing); "
                "face and people detection are disabled. Install opencv-python-headless<5 to enable them.")
# Face/people detectors run far heavier than pixel-diffing; only every Nth tick.
_DETECT_EVERY_N_TICKS = 6


class WatchArea(BaseModel):
    x: float = Field(default=0, ge=0, le=0.95)
    y: float = Field(default=0, ge=0, le=0.95)
    width: float = Field(default=1, ge=0.05, le=1)
    height: float = Field(default=1, ge=0.05, le=1)

    @model_validator(mode="after")
    def inside_frame(self):
        if self.x + self.width > 1.00001 or self.y + self.height > 1.00001:
            raise ValueError("Watch area must fit inside the frame.")
        return self


class CpuSettings(BaseModel):
    enabled: bool = True
    motion_alerts: bool = True
    quality_alerts: bool = False
    face_alerts: bool = False
    people_alerts: bool = False
    object_alerts: bool = False
    motion_threshold: float = Field(default=0.02, ge=0.001, le=0.8)
    cooldown_seconds: int = Field(default=30, ge=5, le=3600)
    dark_threshold: int = Field(default=25, ge=1, le=150)
    blur_threshold: float = Field(default=30, ge=1, le=1000)
    object_confidence: float = Field(default=0.4, ge=0.1, le=0.95)
    area: WatchArea = Field(default_factory=WatchArea)


def detect_faces(gray):
    """Haar-cascade face detection on a 320x180 grayscale frame. Returns fractional boxes."""
    if not _OBJDETECT_AVAILABLE:
        return []
    found = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
    return [[fx / 320, fy / 180, fw / 320, fh / 180] for fx, fy, fw, fh in found]


def detect_people(small_bgr):
    """HOG+SVM pedestrian detection. Runs on the same 320x180 frame as other measurements."""
    if not _OBJDETECT_AVAILABLE:
        return []
    found, _ = _HOG.detectMultiScale(small_bgr, winStride=(4, 4), padding=(8, 8), scale=1.05)
    return [[px / 320, py / 180, pw / 320, ph / 180] for px, py, pw, ph in found]


def measure(frame, previous, settings, run_detectors=False):
    """Measurements at a fixed 320×180 resolution; boxes use full-frame coordinates."""
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    area = settings.area
    x, y = int(area.x * 320), int(area.y * 180)
    x2, y2 = min(320, int((area.x + area.width) * 320)), min(180, int((area.y + area.height) * 180))
    crop = gray[y:y2, x:x2]
    brightness = float(crop.mean())
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    mask = np.zeros_like(gray)
    if previous is not None:
        delta = cv2.absdiff(blurred, previous)
        _, binary = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask[y:y2, x:x2] = binary[y:y2, x:x2]
    fraction = float(np.count_nonzero(mask)) / crop.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        if cv2.contourArea(contour) < 20:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        boxes.append([bx / 320, by / 180, bw / 320, bh / 180])
    result = {
        "motion_percent": round(fraction * 100, 2),
        "motion": previous is not None and fraction >= settings.motion_threshold,
        "brightness": round(brightness, 1), "sharpness": round(sharpness, 1),
        "low_light": brightness < settings.dark_threshold,
        "low_detail": sharpness < settings.blur_threshold,
        "boxes": boxes, "warming_up": previous is None,
    }
    if run_detectors and (settings.face_alerts or settings.people_alerts):
        face_boxes = detect_faces(gray) if settings.face_alerts else []
        people_boxes = detect_people(small) if settings.people_alerts else []
        result.update({
            "face_count": len(face_boxes), "face_boxes": face_boxes,
            "people_count": len(people_boxes), "people_boxes": people_boxes,
        })
    if run_detectors and settings.object_alerts:
        # Full-resolution frame, not the 320x180 downscale used above — YOLO
        # needs real detail, and it isn't cheap enough to run every tick anyway.
        objects = yolo.detect(frame, conf_threshold=settings.object_confidence)
        result["objects"] = [
            {"class": name, "confidence": conf, "box": box} for name, conf, box in objects
        ]
    return result, blurred


def annotate(frame, result, settings):
    image = frame.copy()
    height, width = image.shape[:2]
    area = settings.area
    rectangles = [[area.x, area.y, area.width, area.height]] + result.get("boxes", [])
    for x, y, w, h in rectangles:
        start, end = (int(x * width), int(y * height)), (int((x + w) * width) - 1, int((y + h) * height) - 1)
        cv2.rectangle(image, start, end, (0, 0, 0), 5)
        cv2.rectangle(image, start, end, (255, 255, 255), 2)
    for boxes_key in ("face_boxes", "people_boxes"):
        for x, y, w, h in result.get(boxes_key, []):
            start, end = (int(x * width), int(y * height)), (int((x + w) * width) - 1, int((y + h) * height) - 1)
            cv2.rectangle(image, start, end, (0, 0, 0), 4)
            cv2.rectangle(image, start, end, (255, 255, 255), 1)
    for obj in result.get("objects", []):
        x, y, w, h = obj["box"]
        start, end = (int(x * width), int(y * height)), (int((x + w) * width) - 1, int((y + h) * height) - 1)
        cv2.rectangle(image, start, end, (0, 0, 0), 4)
        cv2.rectangle(image, start, end, (255, 255, 255), 1)
        label = f"{obj['class']} {obj['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_start = (start[0], max(0, start[1] - th - 6))
        cv2.rectangle(image, label_start, (start[0] + tw + 6, start[1]), (0, 0, 0), -1)
        cv2.putText(image, label, (start[0] + 3, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


class CpuMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._states = {}
        self._frames = {}
        self._settings = {}
        self._generation = {}
        self._last_pruned = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="argus-cpu")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def reset(self, camera_id):
        with self._lock:
            self._states.pop(camera_id, None)
            self._frames.pop(camera_id, None)
            self._generation[camera_id] = self._generation.get(camera_id, 0) + 1

    def status(self, camera_id):
        with self._lock:
            state = self._states.get(camera_id)
            public = dict(state["result"]) if state else {"warming_up": True}
        if not registry.is_online(camera_id):
            return {"online": False, "warming_up": False, "message": "Source offline; no current measurements."}
        if public.get("sampled_at") and time.time() - public["sampled_at"] > 5:
            return {"online": True, "warming_up": True, "message": "Measurements are stale; waiting for the CPU worker."}
        return {"online": True, **public}

    def snapshot(self, camera_id):
        if not registry.is_online(camera_id):
            return None
        with self._lock:
            entry = self._frames.get(camera_id)
            if entry is None or time.time() - entry[1] > 5:
                return None
            return entry[0]

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.tick()
            except Exception:
                log.exception("CPU analysis tick failed")
            self._stop.wait(max(0.05, 0.5 - (time.monotonic() - started)))

    def _prune_detections(self, db):
        # Runs at most every 10 minutes — a delete query on every 0.5s tick
        # would cost far more than the table it's pruning.
        now = time.monotonic()
        if now - self._last_pruned < 600:
            return
        self._last_pruned = now
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=DETECTION_RETENTION_HOURS)
        db.query(Detection).filter(Detection.created_at < cutoff).delete()
        db.commit()

    def tick(self):
        with SessionLocal() as db:
            self._prune_detections(db)
            cameras = db.query(Camera).all()
            configs = {row.camera_id: row.settings for row in db.query(CpuConfig).all()}
            ids = {camera.id for camera in cameras}
            with self._lock:
                for mapping in (self._states, self._frames, self._settings, self._generation):
                    for key in list(mapping):
                        if key not in ids:
                            mapping.pop(key, None)
            for camera in cameras:
                try:
                    settings = CpuSettings(**configs.get(camera.id, {}))
                    feed = registry.get(camera.id)
                    frame = feed.get_frame() if feed else None
                    if not settings.enabled or frame is None:
                        self.reset(camera.id)
                        continue
                    self.process(db, camera, frame, settings)
                except Exception:
                    db.rollback()
                    log.exception("CPU analysis failed for source %s", camera.id)

    def process(self, db, camera, frame, settings):
        started = time.monotonic()
        with self._lock:
            generation = self._generation.get(camera.id, 0)
            old = self._states.get(camera.id)
            if self._settings.get(camera.id) != settings.model_dump():
                old = None
        previous = old["gray"] if old else None
        tick = (old["tick"] + 1) if old else 0
        run_detectors = tick % _DETECT_EVERY_N_TICKS == 0
        result, gray = measure(frame, previous, settings, run_detectors=run_detectors)
        if not run_detectors and old and "result" in old:
            # Carry the last detector reading forward between the heavier detection ticks.
            for key in ("face_count", "face_boxes", "people_count", "people_boxes", "objects"):
                if key in old["result"]:
                    result[key] = old["result"][key]
        streaks = dict(old["streaks"]) if old else {}
        active = dict(old["active"]) if old else {}
        last_event = dict(old["last_event"]) if old else {}
        now = time.monotonic()
        candidates = []
        checks = [
            ("motion", settings.motion_alerts, result["motion"], "CPU · Motion detected"),
            ("low_light", settings.quality_alerts, result["low_light"], "CPU · Low-light frame"),
            ("low_detail", settings.quality_alerts, result["low_detail"], "CPU · Low detail / possible blur"),
        ]
        if run_detectors:
            checks.append(("face", settings.face_alerts, result.get("face_count", 0) > 0, "CPU · Face detected"))
            checks.append(("people", settings.people_alerts, result.get("people_count", 0) > 0, "CPU · Person detected"))
        if run_detectors and settings.object_alerts:
            detected_classes = {obj["class"] for obj in result.get("objects", [])}
            rules = (
                db.query(AlertRule)
                .filter(AlertRule.enabled.is_(True), AlertRule.source == "cpu")
                .filter((AlertRule.camera_id == camera.id) | (AlertRule.camera_id.is_(None)))
                .all()
            )
            for rule in rules:
                flag = f"object:{rule.target}"
                checks.append((flag, True, rule.target in detected_classes, f"CPU · {rule.target.title()} detected"))
        for flag, enabled, detected, category in checks:
            streaks[flag] = streaks.get(flag, 0) + 1 if detected else 0
            quiet_key = flag + "_quiet"
            streaks[quiet_key] = 0 if detected else streaks.get(quiet_key, 0) + 1
            if streaks[quiet_key] >= 4:
                active[flag] = False
            if enabled and streaks[flag] >= 2 and not active.get(flag) and now - last_event.get(flag, -float("inf")) >= settings.cooldown_seconds:
                candidates.append((flag, category))
                active[flag] = True
                last_event[flag] = now
        drawn = annotate(frame, result, settings)
        ok, jpeg = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        result.update({"sampled_at": time.time(), "checked_at": datetime.now(timezone.utc).isoformat(), "analysis_ms": round((time.monotonic() - started) * 1000, 1), "area": settings.area.model_dump()})
        # Dense per-tick log: every time a detector actually ran (not the
        # carried-forward ticks in between), regardless of whether anything
        # changed. Distinct from Event, which only logs state changes.
        log_detection = run_detectors and (settings.face_alerts or settings.people_alerts or settings.object_alerts)
        # A saved configuration invalidates any in-flight result from the old area.
        with self._lock:
            if generation != self._generation.get(camera.id, 0):
                return
            if candidates:
                snapshot = image_to_data_uri(frame_to_pil(drawn))
                for flag, category in candidates:
                    summary = f"OpenCV measurement in the watch area: changed pixels {result['motion_percent']}%; brightness {result['brightness']}/255; edge-detail score {result['sharpness']}; faces {result.get('face_count', 0)}; people {result.get('people_count', 0)}. No identity inferred."
                    severity = "info" if flag in ("motion", "face", "people") or flag.startswith("object:") else "warning"
                    db.add(Event(camera_id=camera.id, severity=severity, category=category, summary=summary, snapshot=snapshot))
            if log_detection:
                db.add(Detection(
                    camera_id=camera.id,
                    face_count=result.get("face_count", 0),
                    people_count=result.get("people_count", 0),
                    objects=[{"class": o["class"], "confidence": o["confidence"]} for o in result.get("objects", [])],
                ))
            if candidates or log_detection:
                db.commit()
            self._states[camera.id] = {"gray": gray, "result": result, "streaks": streaks, "active": active, "last_event": last_event, "tick": tick}
            self._settings[camera.id] = settings.model_dump()
            if ok:
                self._frames[camera.id] = (jpeg.tobytes(), time.time())


def events_csv(events):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "camera_id", "created_at", "severity", "category", "summary"])
    def safe(value):
        text = str(value or "")
        # Prevent spreadsheet formula evaluation in text exported from model output.
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")) else text
    for event in events:
        row = event.to_dict()
        writer.writerow([safe(row[key]) for key in ("id", "camera_id", "created_at", "severity", "category", "summary")])
    return output.getvalue()


cpu_monitor = CpuMonitor()
