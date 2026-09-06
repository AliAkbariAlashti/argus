import asyncio
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

import anyio
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .camera import registry
from .config import (
    CHAT_FRAMES_SINGLE_CAMERA,
    CHAT_HISTORY_TURNS,
    MODEL_ID,
    MOTION_BUFFER_SECONDS,
    SEED_CAMERAS,
    SUGGESTED_PROMPTS,
    VIDEOS_DIR,
)
from .db import Base, engine, get_db
from .grounding import (
    build_fleet_prompt,
    build_grounded_prompt,
    metadata_answer,
    route_question,
)
from .imaging import frame_to_pil, image_to_data_uri
from .models import AlertRule, Camera, ChatMessage, Event
from .monitor import monitor
from .vlm import vlm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwenvl.main")

app = FastAPI(title="Sentinel Vision API")

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


def _seed_and_start_cameras():
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        if db.query(Camera).count() == 0:
            for seed in SEED_CAMERAS:
                db.add(Camera(**seed))
            db.commit()

        for camera in db.query(Camera).all():
            registry.add(camera.id, camera.source_path)
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    _seed_and_start_cameras()
    threading.Thread(target=vlm.load, daemon=True).start()
    monitor.start()


@app.on_event("shutdown")
def on_shutdown():
    registry.stop_all()
    monitor.stop()


@app.get("/api/status")
async def status():
    return {
        "model_ready": vlm.ready,
        "model_error": vlm.error,
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
            "id": MODEL_ID,
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


@app.post("/api/cameras")
async def create_camera(
    name: str = Form(...),
    location: str = Form(""),
    zone_tags: str = Form(""),  # comma-separated
    description: str = Form(""),
    video: Optional[UploadFile] = None,
    db: Session = Depends(get_db),
):
    if video is None:
        raise HTTPException(
            status_code=400,
            detail="A video file is required (IP camera / RTSP connections are coming soon).",
        )

    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}",
        )

    tags = [t.strip() for t in zone_tags.split(",") if t.strip()]

    filename = f"{uuid.uuid4().hex[:10]}{ext}"
    dest = VIDEOS_DIR / filename
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    camera = Camera(
        name=name,
        location=location,
        zone_tags=tags,
        description=description,
        source_type="file",
        source_path=filename,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    registry.add(camera.id, camera.source_path)

    return _camera_out(camera)


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    zone_tags: Optional[list[str]] = None
    description: Optional[str] = None


@app.put("/api/cameras/{cam_id}")
def update_camera(cam_id: str, payload: CameraUpdate, db: Session = Depends(get_db)):
    camera = db.get(Camera, cam_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if payload.name is not None:
        camera.name = payload.name
    if payload.location is not None:
        camera.location = payload.location
    if payload.zone_tags is not None:
        camera.zone_tags = payload.zone_tags
    if payload.description is not None:
        camera.description = payload.description

    db.commit()
    db.refresh(camera)
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
    rows = query.limit(min(limit, 500)).all()
    return [r.to_dict() for r in rows]


# ---------- Alert rules ----------


class AlertRuleCreate(BaseModel):
    camera_id: Optional[str] = None
    target: str


class AlertRuleUpdate(BaseModel):
    enabled: bool


@app.get("/api/alert-rules")
def list_alert_rules(db: Session = Depends(get_db)):
    return [r.to_dict() for r in db.query(AlertRule).order_by(AlertRule.created_at).all()]


@app.post("/api/alert-rules")
def create_alert_rule(payload: AlertRuleCreate, db: Session = Depends(get_db)):
    target = payload.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="A target phrase is required.")
    rule = AlertRule(camera_id=payload.camera_id or None, target=target)
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule.to_dict()


@app.put("/api/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, payload: AlertRuleUpdate, db: Session = Depends(get_db)):
    rule = db.get(AlertRule, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    rule.enabled = payload.enabled
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
    while True:
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


def _primary_camera_index(answer: str, cameras: list[Camera]) -> int:
    """Which camera a fleet answer is primarily about, for the snapshot
    thumbnail — the camera whose name appears earliest in the answer text,
    or the first camera if none is named. Fleet answers are prompted to
    always name the relevant camera(s) (see build_fleet_prompt), so this
    matches the thumbnail to what the text actually says instead of always
    showing whichever camera happened to be first by creation order."""
    answer_lower = answer.lower()
    best_idx, best_pos = 0, None
    for i, cam in enumerate(cameras):
        name = (cam.name or "").lower().strip()
        if not name:
            continue
        pos = answer_lower.find(name)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_idx, best_pos = i, pos
    return best_idx


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
    question: str
    focused_camera_id: Optional[str] = None


@app.get("/api/chat/history")
def chat_history(db: Session = Depends(get_db)):
    rows = db.query(ChatMessage).order_by(ChatMessage.id).all()
    return [r.to_dict() for r in rows]


@app.delete("/api/chat/history")
def clear_chat_history(db: Session = Depends(get_db)):
    deleted = db.query(ChatMessage).delete()
    db.commit()
    return {"deleted": deleted}


@app.post("/api/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    all_cameras = db.query(Camera).order_by(Camera.created_at).all()
    if not all_cameras:
        raise HTTPException(status_code=400, detail="No cameras configured yet.")

    history = _recent_history(db)
    _save_turn(db, "user", req.question)

    scope, camera = route_question(req.question, all_cameras, req.focused_camera_id)
    log.info("Chat scope=%s camera=%s q=%r", scope, camera.name if camera else "-", req.question)

    # Answerable from the database alone — no vision needed, and no chance
    # of the model inventing a number it can't see.
    if scope == "metadata":
        camera_dicts = [_camera_out(c) for c in all_cameras]
        answer = metadata_answer(req.question, camera_dicts)
        used = [c["id"] for c in camera_dicts]
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
            detail=vlm.error or "Model is still loading, please wait a moment.",
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
            answer = vlm.ask(images, prompt, history=history, max_new_tokens=768)
        except Exception as exc:  # noqa: BLE001
            log.exception("Fleet inference failed")
            raise HTTPException(status_code=500, detail=str(exc))

        used = [c.id for c in used_cameras]
        snapshot = image_to_data_uri(images[_primary_camera_index(answer, used_cameras)])
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
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    snapshot = image_to_data_uri(images[-1])
    _save_turn(db, "assistant", answer, cameras_used=[camera.id], snapshot=snapshot)
    return {
        "question": req.question,
        "answer": answer,
        "cameras_used": [camera.id],
        "snapshot": snapshot,
        "scope": scope,
    }


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
