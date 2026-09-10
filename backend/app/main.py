import asyncio
import base64
import logging
import json
import time
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import anyio
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from .camera import registry
from .cpu import CpuSettings, cpu_monitor, events_csv, health as cpu_health
from .config import (
    CHAT_FRAMES_SINGLE_CAMERA,
    CHAT_HISTORY_TURNS,
    MODEL_ID,
    MOTION_BUFFER_SECONDS,
    SEED_CAMERAS,
    SUGGESTED_PROMPTS,
    VIDEOS_DIR,
)
from .db import Base, engine, get_db, SessionLocal
from .grounding import (
    build_fleet_prompt,
    build_grounded_prompt,
    extract_primary_cameras,
    is_metadata_question,
    metadata_answer,
    route_question,
)
from .imaging import frame_to_pil, image_to_data_uri
from .models import AlertRule, Camera, ChatMessage, Event, Observation, CpuConfig
from .observation_query import answer_observation_question
from .visual_search import find_similar
from .monitor import monitor
from .runtime import vlm
from . import yolo
from . import history

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwenvl.main")

app = FastAPI(title="Argus Video Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _raise_threadpool_capacity():
    # Sync (blocking) routes below — DB calls, cv2, model inference, file
    # copies — run in Starlette's shared threadpool. The default cap is low
    # enough that a handful of dashboard tiles polling snapshots/status can
    # starve simple CRUD requests. This is a plain concurrency ceiling, not
    # per-request timeout, so raise it well above what the UI can realistically
    # open at once.
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 200


def _add_alert_rule_columns():
    # No migration tool in this project yet; create_all only adds missing
    # tables, not columns on an already-existing one.
    with engine.begin() as conn:
        existing = {row[0] for row in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'alert_rules'"
        ))}
        if existing and "source" not in existing:
            conn.execute(text("ALTER TABLE alert_rules ADD COLUMN source VARCHAR NOT NULL DEFAULT 'vlm'"))
        if existing and "severity" not in existing:
            conn.execute(text("ALTER TABLE alert_rules ADD COLUMN severity VARCHAR NOT NULL DEFAULT 'info'"))


def _add_event_confidence_column():
    with engine.begin() as conn:
        existing = {row[0] for row in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'events'"
        ))}
        if existing and "confidence" not in existing:
            conn.execute(text("ALTER TABLE events ADD COLUMN confidence INTEGER"))


def _seed_and_start_cameras():
    Base.metadata.create_all(bind=engine)
    _add_alert_rule_columns()
    _add_event_confidence_column()
    db = next(get_db())
    try:
        if db.query(Camera).count() == 0:
            for seed in SEED_CAMERAS:
                db.add(Camera(**seed))
            db.commit()

        for camera in db.query(Camera).all():
            registry.add(camera.id, camera.source_path, camera.source_type)
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    _seed_and_start_cameras()
    threading.Thread(target=vlm.load, daemon=True).start()
    monitor.start()
    cpu_monitor.start()


@app.on_event("shutdown")
def on_shutdown():
    cpu_monitor.stop()
    registry.stop_all()
    monitor.stop()


@app.get("/api/status")
async def status():
    return {
        "model_ready": vlm.ready,
        "model_error": vlm.error,
        "configured": vlm.configured,
        "provider": vlm.public_settings()["provider"],
    }


@app.get("/api/prompts")
async def suggested_prompts():
    return SUGGESTED_PROMPTS


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    """Operational snapshot: model state, GPU memory, per-camera decode rate."""
    gpu = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_mb": round(total_b / 1024 / 1024),
                "used_mb": round((total_b - free_b) / 1024 / 1024),
                "free_mb": round(free_b / 1024 / 1024),
            }
    except Exception:  # noqa: BLE001
        # torch missing or CUDA unavailable — report "no GPU", not an error.
        log.debug("GPU stats unavailable", exc_info=True)

    cameras = []
    for cam in db.query(Camera).order_by(Camera.created_at).all():
        feed = registry.get(cam.id)
        cameras.append(
            {
                "id": cam.id,
                "name": cam.name,
                "online": registry.is_online(cam.id),
                "fps": feed.get_fps() if feed else 0.0,
            }
        )

    return {
        "model": {
            "id": vlm.public_settings().get("model") or (MODEL_ID if vlm.public_settings()["provider"] == "embedded" else "Not configured"),
            "ready": vlm.ready,
            "error": vlm.error,
        },
        "gpu": gpu,
        "cameras": cameras,
    }


# ---------- Camera CRUD ----------


def _camera_out(camera: Camera) -> dict:
    return {**camera.to_dict(), "online": registry.is_online(camera.id)}


@app.get("/api/cameras")
def list_cameras(db: Session = Depends(get_db)):
    return [_camera_out(c) for c in db.query(Camera).order_by(Camera.created_at).all()]


@app.get("/api/cameras/{cam_id}")
def get_camera(cam_id: str, db: Session = Depends(get_db)):
    camera = db.get(Camera, cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_out(camera)


ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _validate_rtsp_url(value: str) -> str:
    value = value.strip()
    parsed = None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        parsed.port  # also reject malformed or out-of-range ports
    except ValueError:
        hostname = None
    if parsed is None or parsed.scheme not in ("rtsp", "rtsps") or not hostname:
        raise HTTPException(400, "Enter a valid RTSP URL, for example rtsp://192.168.1.50:8554/camera.")
    if any(char.isspace() for char in value):
        raise HTTPException(400, "The RTSP URL cannot contain spaces.")
    return value


@app.post("/api/cameras")
async def create_camera(
    name: str = Form(...),
    location: str = Form(""),
    zone_tags: str = Form(""),  # comma-separated
    description: str = Form(""),
    source_type: str = Form("file"),
    rtsp_url: str = Form(""),
    video: Optional[UploadFile] = None,
    db: Session = Depends(get_db),
):
    tags = [t.strip() for t in zone_tags.split(",") if t.strip()]

    if source_type == "rtsp":
        source_path = _validate_rtsp_url(rtsp_url)
    elif source_type == "file":
        if video is None:
            raise HTTPException(status_code=400, detail="Choose a video file or switch the source type to RTSP.")
        ext = Path(video.filename or "").suffix.lower()
        if ext not in ALLOWED_VIDEO_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}",
            )
        filename = f"{uuid.uuid4().hex[:10]}{ext}"
        dest = VIDEOS_DIR / filename
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        source_path = filename
    else:
        raise HTTPException(status_code=400, detail="Source type must be file or rtsp.")

    camera = Camera(
        name=name,
        location=location,
        zone_tags=tags,
        description=description,
        source_type=source_type,
        source_path=source_path,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    registry.add(camera.id, camera.source_path, camera.source_type)

    return _camera_out(camera)


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    zone_tags: Optional[list[str]] = None
    description: Optional[str] = None
    source_type: Optional[str] = Field(default=None, pattern="^(file|rtsp)$")
    source_path: Optional[str] = Field(default=None, max_length=2000)


@app.put("/api/cameras/{cam_id}")
def update_camera(cam_id: str, payload: CameraUpdate, db: Session = Depends(get_db)):
    camera = db.get(Camera, cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if payload.source_type is not None and payload.source_type != camera.source_type:
        raise HTTPException(400, "Create a new source to change between uploaded video and RTSP.")
    next_type = payload.source_type or camera.source_type
    next_path = payload.source_path if payload.source_path is not None else camera.source_path
    if next_type == "rtsp":
        next_path = _validate_rtsp_url(next_path)
    elif payload.source_type == "file" and payload.source_path is not None:
        raise HTTPException(400, "Uploaded video files cannot be replaced through this form.")
    source_changed = next_type != camera.source_type or next_path != camera.source_path

    if payload.name is not None:
        camera.name = payload.name
    if payload.location is not None:
        camera.location = payload.location
    if payload.zone_tags is not None:
        camera.zone_tags = payload.zone_tags
    if payload.description is not None:
        camera.description = payload.description
    if source_changed:
        registry.remove(cam_id)
        camera.source_type = next_type
        camera.source_path = next_path

    db.commit()
    db.refresh(camera)
    if source_changed:
        registry.add(camera.id, camera.source_path, camera.source_type)
    return _camera_out(camera)


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str, db: Session = Depends(get_db)):
    camera = db.get(Camera, cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    registry.remove(cam_id)

    if camera.source_type == "file":
        video_path = VIDEOS_DIR / camera.source_path
        video_path.unlink(missing_ok=True)

    db.query(CpuConfig).filter(CpuConfig.camera_id == cam_id).delete()
    cpu_monitor.reset(cam_id)
    db.delete(camera)
    db.commit()
    return {"deleted": cam_id}


# ---------- Events (background monitor) ----------


@app.get("/api/events")
def list_events(
    limit: int = 100,
    camera_id: Optional[str] = None,
    severity: Optional[str] = None,
    q: Optional[str] = None,
    since: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Event).order_by(Event.id.desc())
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if severity:
        query = query.filter(Event.severity == severity)
    if q:
        like = f"%{q}%"
        query = query.filter(Event.summary.ilike(like) | Event.category.ilike(like))
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(status_code=400, detail="since must be an ISO-8601 timestamp.")
        # created_at is stored naive-UTC (see models._utc_iso); strip any offset to match.
        query = query.filter(Event.created_at >= since_dt.replace(tzinfo=None))
    rows = query.limit(min(limit, 500)).all()
    return [r.to_dict() for r in rows]


def _observation_time(value: str, field: str):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, f"{field} must be an ISO-8601 timestamp.") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@app.get("/api/observations")
def list_observations(
    camera_id: Optional[str] = None,
    object_type: Optional[str] = None,
    kind: Optional[str] = None,
    action: Optional[str] = None,
    track_id: Optional[str] = None,
    producer_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Indexed structured facts for deterministic and chatbot tool queries."""
    query = db.query(Observation).order_by(Observation.observed_at.desc(), Observation.id.desc())
    if camera_id:
        query = query.filter(Observation.camera_id == camera_id)
    if object_type:
        query = query.filter(Observation.object_type == object_type.strip().lower())
    if kind:
        query = query.filter(Observation.kind == kind)
    if action:
        query = query.filter(Observation.action == action)
    if track_id:
        query = query.filter(Observation.track_id == track_id)
    if producer_id:
        query = query.filter(Observation.producer_id == producer_id)
    if since:
        query = query.filter(Observation.observed_at >= _observation_time(since, "since"))
    if until:
        query = query.filter(Observation.observed_at < _observation_time(until, "until"))
    rows = query.limit(max(1, min(limit, 500))).all()
    return [row.to_dict() for row in rows]


@app.get("/api/observations/ask")
def ask_observations(question: str, camera_id: Optional[str] = None, db: Session = Depends(get_db)):
    cameras = db.query(Camera).order_by(Camera.created_at).all()
    result = answer_observation_question(db, question.strip(), cameras, forced_camera_id=camera_id)
    if result is None:
        raise HTTPException(400, "Ask a historical detection, tracking, crossing, or dwell question.")
    return {"question": question, **result}


@app.get("/api/observations/{observation_id}")
def get_observation(observation_id: str, db: Session = Depends(get_db)):
    row = db.get(Observation, observation_id)
    if row is None:
        raise HTTPException(404, "Observation not found")
    return row.to_dict()


@app.get("/api/visual-search/{observation_id}")
def search_similar_observations(
    observation_id: str,
    camera_id: Optional[str] = None,
    object_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """Find observations with a similar indexed object crop."""
    result = find_similar(
        db,
        observation_id,
        camera_id=camera_id,
        object_type=object_type,
        since=_observation_time(since, "since") if since else None,
        until=_observation_time(until, "until") if until else None,
        limit=limit,
    )
    if result is None:
        raise HTTPException(404, "No visual index exists for this observation.")
    return result


@app.get("/api/events/ask")
def ask_events_history(question: str, db: Session = Depends(get_db)):
    """Rule-based natural-language search over logged events — no model call.
    'When did you last see a person on camera 1' resolves to a plain
    filtered query, since every event already carries camera/category/time."""
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")
    cameras = [{"id": c.id, "name": c.name, "zone_tags": c.zone_tags} for c in db.query(Camera).all()]
    parsed = history.parse_question(question, cameras)

    query = db.query(Event).order_by(Event.id.desc())
    if parsed["camera_id"]:
        query = query.filter(Event.camera_id == parsed["camera_id"])
    if parsed["severity"]:
        query = query.filter(Event.severity == parsed["severity"])
    if parsed["q"]:
        like = f"%{parsed['q']}%"
        query = query.filter(Event.summary.ilike(like) | Event.category.ilike(like))
    if parsed["since"]:
        query = query.filter(Event.created_at >= datetime.fromisoformat(parsed["since"]).replace(tzinfo=None))
    rows = query.limit(parsed["limit"]).all()

    camera_names = {c.id: c.name for c in db.query(Camera).all()}
    events = [r.to_dict() for r in rows]
    if not events:
        answer = "No matching events found."
    elif parsed["limit"] == 1:
        row = events[0]
        answer = f"Last seen: {row['category']} on {camera_names.get(row['camera_id'], row['camera_id'])}, {row['created_at']}."
    else:
        answer = f"{len(events)} matching event(s)."

    return {"question": question, "parsed": parsed, "answer": answer, "events": events}


@app.get("/api/events/export")
def export_events(camera_id: Optional[str] = None, severity: Optional[str] = None, q: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Event).order_by(Event.id.desc())
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)
    if severity:
        query = query.filter(Event.severity == severity)
    if q:
        query = query.filter(Event.summary.ilike(f"%{q}%") | Event.category.ilike(f"%{q}%"))
    content = events_csv(query.limit(5000).all())
    return Response(content=content, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="argus-events.csv"'})


@app.get("/api/cpu")
def cpu_status(db: Session = Depends(get_db)):
    configs = {row.camera_id: row.settings for row in db.query(CpuConfig).all()}
    return [{"camera_id": cam.id, "name": cam.name,
             "settings": CpuSettings(**configs.get(cam.id, {})).model_dump(),
             "status": cpu_monitor.status(cam.id)} for cam in db.query(Camera).order_by(Camera.created_at).all()]


@app.get("/api/cpu/health")
def cpu_dependencies_health():
    return cpu_health()


@app.put("/api/cpu/{camera_id}")
def configure_cpu(camera_id: str, payload: CpuSettings, db: Session = Depends(get_db)):
    if db.get(Camera, camera_id) is None:
        raise HTTPException(404, "Camera not found")
    row = db.get(CpuConfig, camera_id)
    if row is None:
        row = CpuConfig(camera_id=camera_id)
        db.add(row)
    row.settings = payload.model_dump()
    db.commit()
    cpu_monitor.reset(camera_id)
    return row.settings


@app.get("/api/cpu/{camera_id}/snapshot")
def cpu_snapshot(camera_id: str):
    jpeg = cpu_monitor.snapshot(camera_id)
    if jpeg is None:
        raise HTTPException(503, "No current CPU analysis frame. Enable CPU tools and wait for a sample.")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------- Alert rules ----------


class AlertRuleCreate(BaseModel):
    camera_id: Optional[str] = None
    source: str = Field(default="vlm", pattern="^(vlm|cpu)$")
    target: str
    severity: Optional[str] = Field(default=None, pattern="^(info|warning|critical)$")


class AlertRuleUpdate(BaseModel):
    enabled: Optional[bool] = None
    severity: Optional[str] = Field(default=None, pattern="^(info|warning|critical)$")


@app.get("/api/alert-rules")
def list_alert_rules(db: Session = Depends(get_db)):
    return [r.to_dict() for r in db.query(AlertRule).order_by(AlertRule.created_at).all()]


@app.get("/api/alert-rules/cpu-classes")
def alert_rule_cpu_classes():
    """The fixed vocabulary CPU-sourced rules can match — YOLO's class list,
    plus a suggested default severity per class the UI can pre-fill."""
    return {"classes": yolo.CLASSES, "default_severity": yolo.DEFAULT_SEVERITY, "available": yolo.available()}


@app.post("/api/alert-rules")
def create_alert_rule(payload: AlertRuleCreate, db: Session = Depends(get_db)):
    target = payload.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="A target phrase is required.")
    if payload.source == "cpu" and target.lower() not in yolo.CLASSES:
        raise HTTPException(status_code=400, detail=f'"{target}" is not one of the CPU object classes. Choose from the list.')
    target = target.lower() if payload.source == "cpu" else target
    severity = payload.severity or (yolo.DEFAULT_SEVERITY.get(target, "info") if payload.source == "cpu" else "info")
    rule = AlertRule(camera_id=payload.camera_id or None, source=payload.source, target=target, severity=severity)
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule.to_dict()


@app.put("/api/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, payload: AlertRuleUpdate, db: Session = Depends(get_db)):
    rule = db.get(AlertRule, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    if payload.enabled is not None:
        rule.enabled = payload.enabled
    if payload.severity is not None:
        if rule.source != "cpu":
            raise HTTPException(status_code=400, detail="Only CPU-sourced rules have an adjustable severity.")
        rule.severity = payload.severity
    db.commit()
    db.refresh(rule)
    return rule.to_dict()


@app.delete("/api/alert-rules/{rule_id}")
def delete_alert_rule(rule_id: str, db: Session = Depends(get_db)):
    rule = db.get(AlertRule, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    db.delete(rule)
    db.commit()
    return {"deleted": rule_id}


# ---------- Streaming ----------


async def _mjpeg_generator(cam_id: str):
    # Async generator: sleeps here yield control back to the event loop
    # instead of pinning a threadpool worker for the whole connection
    # lifetime, which matters a lot for a stream that stays open for
    # as long as a browser tab is on this camera.
    feed = registry.get(cam_id)
    if feed is None:
        return
    boundary = b"--frame"
    while feed.running and registry.get(cam_id) is feed:
        jpeg = feed.get_jpeg()
        if jpeg is not None:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        await asyncio.sleep(1 / 25)


@app.get("/api/cameras/{cam_id}/stream")
async def camera_stream(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras/{cam_id}/snapshot")
async def camera_snapshot(cam_id: str):
    feed = registry.get(cam_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    jpeg = feed.get_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Camera has no frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


# ---------- Chat ----------
#
# One global chat, not one per camera: every question is answered with
# whatever context it actually needs, decided per-message —
#   1. fleet metadata questions ("how many cameras") -> answered directly
#      from the DB, no VLM call, so the answer is always correct.
#   2. fleet vision questions ("is anyone in the building") -> every
#      camera's current frame is sent to the VLM in one call, each frame
#      labeled by camera name.
#   3. a specific camera mentioned by name/zone/tag -> that camera's frame.
#   4. otherwise -> falls back to whichever camera the user currently has
#      focused in the UI (if any), so the per-camera prompt chips still do
#      the intuitive thing.


def _frame_to_image(cam_id: str):
    feed = registry.get(cam_id)
    frame = feed.get_frame() if feed else None
    if frame is None:
        return None
    return frame_to_pil(frame)


def _frame_sequence_to_images(cam_id: str, count: int):
    """Recent frames for one camera, so the model can see movement."""
    feed = registry.get(cam_id)
    if feed is None:
        return []
    return [frame_to_pil(f) for f in feed.get_frame_sequence(count)]


def _recent_history(db: Session) -> list[dict]:
    """Prior turns, oldest first, for replay as model context."""
    rows = (
        db.query(ChatMessage)
        .order_by(ChatMessage.id.desc())
        .limit(CHAT_HISTORY_TURNS)
        .all()
    )
    return [{"role": r.role, "text": r.text} for r in reversed(rows)]


def _save_turn(db: Session, role: str, text: str, cameras_used=None, snapshot=None):
    msg = ChatMessage(
        role=role,
        text=text,
        cameras_used=cameras_used or [],
        snapshot=snapshot,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scope: str = Field(default="auto", pattern="^(auto|camera|fleet|cpu)$")
    focused_camera_id: Optional[str] = None


@app.get("/api/chat/history")
def chat_history(db: Session = Depends(get_db)):
    rows = db.query(ChatMessage).order_by(ChatMessage.id).all()
    return [r.to_dict() for r in rows]


@app.delete("/api/chat/history")
def clear_chat_history(db: Session = Depends(get_db)):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for the current answer before clearing the conversation.")
    try:
        deleted = db.query(ChatMessage).delete()
        db.commit()
        return {"deleted": deleted}
    finally:
        _chat_lock.release()


def _answer_chat(req: ChatRequest, db: Session, on_token=None):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    all_cameras = db.query(Camera).order_by(Camera.created_at).all()
    if not all_cameras:
        raise HTTPException(status_code=400, detail="No cameras configured yet.")

    if req.scope == "cpu":
        selected = [c for c in all_cameras if c.id == req.focused_camera_id] if req.focused_camera_id else all_cameras
        if not selected:
            raise HTTPException(400, "Select an available source.")
        lines = ["CPU measurements (OpenCV; no vision model):"]
        snapshot = None
        for cam in selected:
            config = db.get(CpuConfig, cam.id)
            enabled = CpuSettings(**(config.settings if config else {})).enabled
            state = cpu_monitor.status(cam.id)
            if not enabled:
                lines.append(f"{cam.name}: CPU analysis is disabled.")
            elif state.get("warming_up") or not state.get("online"):
                lines.append(f"{cam.name}: {state.get('message', 'Waiting for the first CPU samples.')}")
            else:
                lines.append(f"{cam.name}: motion {'above' if state['motion'] else 'below'} the configured trigger; {state['motion_percent']}% changed pixels in the watch area. Brightness {state['brightness']}/255; edge-detail score {state['sharpness']}. Sample time: {state['checked_at']}.")
                if state["low_light"]:
                    lines.append("Low-light indicator is active.")
                if state["low_detail"]:
                    lines.append("Low-detail indicator is active; this can be blur or a plain scene.")
                if snapshot is None:
                    jpeg = cpu_monitor.snapshot(cam.id)
                    if jpeg:
                        snapshot = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        lines.append("These measurements do not identify people, objects, or behaviour. Choose a vision-model mode for scene descriptions.")
        answer = "\n\n".join(lines)
        used = [c.id for c in selected]
        _save_turn(db, "user", req.question)
        _save_turn(db, "assistant", answer, cameras_used=used, snapshot=snapshot)
        return {"question": req.question, "answer": answer, "scope": "cpu", "cameras_used": used, "snapshot": snapshot}

    forced_camera_id = None
    if req.scope == "camera":
        focused_camera = next((camera for camera in all_cameras if camera.id == req.focused_camera_id), None)
        if focused_camera is None and not is_metadata_question(req.question):
            raise HTTPException(status_code=400, detail="Select a camera first.")
        if focused_camera is not None:
            forced_camera_id = focused_camera.id
    observation_answer = answer_observation_question(db, req.question, all_cameras, forced_camera_id)
    if observation_answer is not None:
        _save_turn(db, "user", req.question)
        _save_turn(
            db, "assistant", observation_answer["answer"],
            cameras_used=observation_answer["cameras_used"], snapshot=observation_answer["snapshot"],
        )
        return {
            "question": req.question,
            "answer": observation_answer["answer"],
            "cameras_used": observation_answer["cameras_used"],
            "snapshot": observation_answer["snapshot"],
            "scope": "observations",
            "query_plan": observation_answer["plan"],
        }

    history = _recent_history(db)

    scope, camera = route_question(req.question, all_cameras, req.focused_camera_id)
    if req.scope == "fleet" and scope != "metadata":
        scope, camera = "fleet", None
    elif req.scope == "camera" and scope != "metadata":
        camera = next((c for c in all_cameras if c.id == req.focused_camera_id), None)
        if camera is None:
            raise HTTPException(status_code=400, detail="Select a camera first.")
        scope = "camera"
    log.info("Chat scope=%s camera=%s q=%r", scope, camera.name if camera else "-", req.question)

    # Answerable from the database alone — no vision needed, and no chance
    # of the model inventing a number it can't see.
    if scope == "metadata":
        camera_dicts = [_camera_out(c) for c in all_cameras]
        answer = metadata_answer(req.question, camera_dicts)
        used = [c["id"] for c in camera_dicts]
        _save_turn(db, "user", req.question)
        _save_turn(db, "assistant", answer, cameras_used=used)
        return {
            "question": req.question,
            "answer": answer,
            "cameras_used": used,
            "scope": scope,
        }

    if not vlm.ready:
        raise HTTPException(
            status_code=503,
            detail=vlm.error or "Open AI setup to configure and test a vision model. Camera viewing works without AI.",
        )

    # One current frame from every camera that has one.
    if scope == "fleet":
        images = []
        used_cameras = []
        for cam in all_cameras:
            img = _frame_to_image(cam.id)
            if img is not None:
                images.append(img)
                used_cameras.append(cam)

        if not images:
            raise HTTPException(status_code=503, detail="No cameras have a frame yet.")

        prompt = build_fleet_prompt(used_cameras, req.question)
        try:
            raw_answer = vlm.ask(images, prompt, history=history, max_new_tokens=768)
        except Exception as exc:  # noqa: BLE001
            log.exception("Fleet inference failed")
            raise HTTPException(status_code=500, detail=str(exc))

        # The model names which camera(s) its answer is actually about via a
        # trailing [[cameras: ...]] tag (see build_fleet_prompt) instead of
        # us guessing from where a name happens to appear in the prose — a
        # well-behaved answer also names the clear cameras, which broke that
        # heuristic. Falls back to the first camera if the tag is missing,
        # unparseable, or names nothing we recognize.
        answer, primary_cameras = extract_primary_cameras(raw_answer, used_cameras)
        snapshot_camera = primary_cameras[0] if primary_cameras else used_cameras[0]
        snapshot_index = used_cameras.index(snapshot_camera)

        used = [c.id for c in used_cameras]
        snapshot = image_to_data_uri(images[snapshot_index])
        _save_turn(db, "user", req.question)
        _save_turn(db, "assistant", answer, cameras_used=used, snapshot=snapshot)
        return {
            "question": req.question,
            "answer": answer,
            "cameras_used": used,
            "snapshot": snapshot,
            "scope": scope,
        }

    # A camera the user named, or the one they're looking at.
    images = _frame_sequence_to_images(camera.id, CHAT_FRAMES_SINGLE_CAMERA)
    if not images:
        raise HTTPException(status_code=503, detail=f"Camera '{camera.name}' has no frame yet.")

    grounded_question = build_grounded_prompt(camera, req.question, frame_count=len(images))

    try:
        answer = vlm.ask(
            images,
            grounded_question,
            history=history,
            fps=1.0 / MOTION_BUFFER_SECONDS,
            on_token=on_token,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    snapshot = image_to_data_uri(images[-1])
    _save_turn(db, "user", req.question)
    _save_turn(db, "assistant", answer, cameras_used=[camera.id], snapshot=snapshot)
    return {
        "question": req.question,
        "answer": answer,
        "cameras_used": [camera.id],
        "snapshot": snapshot,
        "scope": scope,
    }


# One conversation and one GPU: reject overlapping questions instead of
# silently building an unbounded queue or interleaving conversation turns.
_chat_lock = threading.Lock()


@app.post("/api/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="An answer is already in progress. Please wait.")
    vlm.interactive.set()
    try:
        return _answer_chat(req, db)
    finally:
        vlm.interactive.clear()
        _chat_lock.release()


@app.post("/api/chat/stream")
async def stream_chat(req: ChatRequest):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="An answer is already in progress. Please wait.")
    vlm.interactive.set()
    loop = asyncio.get_running_loop()
    messages = asyncio.Queue()
    disconnected = threading.Event()

    def emit(kind, payload):
        if not disconnected.is_set() and not loop.is_closed():
            loop.call_soon_threadsafe(messages.put_nowait, (kind, payload))

    def work():
        started = time.monotonic()
        try:
            with SessionLocal() as db:
                result = _answer_chat(req, db, on_token=lambda token: emit("token", {"text": token}))
            result["elapsed_seconds"] = round(time.monotonic() - started, 1)
            emit("answer", result)
        except HTTPException as exc:
            emit("error", {"detail": exc.detail})
        except Exception:
            log.exception("Streaming chat failed")
            emit("error", {"detail": "Analysis failed. Check system health and try again."})
        finally:
            vlm.interactive.clear()
            _chat_lock.release()
            emit("done", {})

    async def events():
        try:
            yield 'event: status\ndata: {"text":"Planning the query and gathering evidence…"}\n\n'
            while True:
                try:
                    kind, payload = await asyncio.wait_for(messages.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
                    continue
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
                if kind == "done":
                    break
        finally:
            disconnected.set()

    # The worker owns its DB session and lock through completion, even if
    # the browser disconnects. It never holds a request-scoped session.
    try:
        threading.Thread(target=work, daemon=True).start()
    except Exception:
        vlm.interactive.clear()
        _chat_lock.release()
        raise
    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


class RuntimeSettings(BaseModel):
    provider: str = Field(pattern="^(none|compatible|embedded)$")
    base_url: str = Field(default="", max_length=500)
    model: str = Field(default="", max_length=200)
    api_key: Optional[str] = Field(default=None, max_length=1000)
    profile_id: Optional[str] = Field(default=None, max_length=80)
    name: Optional[str] = Field(default=None, max_length=100)
    create_new: bool = False
    activate: bool = True
    allow_remote: bool = False
    monitor_enabled: bool = False


@app.get("/api/runtime")
def get_runtime():
    from .runtime import hardware_profile
    return {"settings": vlm.public_settings(), "hardware": hardware_profile(), "profiles": vlm.public_profiles(), "active_profile_id": vlm.public_settings().get("profile_id")}


@app.put("/api/runtime")
def configure_runtime(payload: RuntimeSettings):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for the current answer before changing AI settings.")
    try:
        return vlm.configure(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        _chat_lock.release()


@app.post("/api/runtime/profiles/{profile_id}/activate")
def activate_runtime_profile(profile_id: str):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for the current answer before changing AI settings.")
    try:
        try:
            return vlm.activate(profile_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
    finally:
        _chat_lock.release()


@app.delete("/api/runtime/profiles/{profile_id}")
def delete_runtime_profile(profile_id: str):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for the current answer before changing AI settings.")
    try:
        try:
            return vlm.delete_profile(profile_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
    finally:
        _chat_lock.release()


@app.get("/api/runtime/usage")
def runtime_usage(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    rows = db.query(ChatMessage).filter(ChatMessage.role == "assistant").all()
    day_start = now - timedelta(days=6)
    counts = {}
    for offset in range(7):
        day = (now - timedelta(days=6 - offset)).date()
        counts[day.isoformat()] = 0
    for row in rows:
        created = row.created_at
        if created is None:
            continue
        created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
        if created >= day_start:
            key = created.date().isoformat()
            if key in counts:
                counts[key] += 1
    recent_24h = 0
    recent_7d = 0
    for row in rows:
        if row.created_at is None:
            continue
        created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
        recent_24h += created >= now - timedelta(days=1)
        recent_7d += created >= now - timedelta(days=7)
    last_response = max((row.created_at for row in rows if row.created_at), default=None)
    if last_response and not last_response.tzinfo:
        last_response = last_response.replace(tzinfo=timezone.utc)
    return {
        "total_requests": len(rows),
        "last_24h": recent_24h,
        "last_7d": recent_7d,
        "last_response_at": last_response.isoformat() if last_response else None,
        "daily": [{"date": key, "requests": value} for key, value in counts.items()],
    }


@app.post("/api/runtime/test")
def test_runtime():
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "An answer or connection test is already running.")
    vlm.interactive.set()
    try:
        return vlm.test()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        vlm.interactive.clear()
        _chat_lock.release()


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
