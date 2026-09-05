import asyncio
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

import anyio
import cv2
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .camera import registry
from .config import SEED_CAMERAS, SUGGESTED_PROMPTS, VIDEOS_DIR
from .db import Base, engine, get_db
from .grounding import build_grounded_prompt
from .models import Camera
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


@app.on_event("shutdown")
def on_shutdown():
    registry.stop_all()


@app.get("/api/status")
async def status():
    return {
        "model_ready": vlm.ready,
        "model_error": vlm.error,
    }


@app.get("/api/prompts")
async def suggested_prompts():
    return SUGGESTED_PROMPTS


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


class ChatRequest(BaseModel):
    camera_id: str
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not vlm.ready:
        raise HTTPException(
            status_code=503,
            detail=vlm.error or "Model is still loading, please wait a moment.",
        )

    camera = db.get(Camera, req.camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    feed = registry.get(req.camera_id)
    frame = feed.get_frame() if feed else None
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera has no frame yet")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    grounded_question = build_grounded_prompt(camera, req.question)

    try:
        answer = vlm.ask(image, grounded_question)
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"camera_id": req.camera_id, "question": req.question, "answer": answer}


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
