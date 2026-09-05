import cv2
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from .config import CAMERAS, SUGGESTED_PROMPTS
from .camera import registry
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
def on_startup():
    registry.start_all()
    threading.Thread(target=vlm.load, daemon=True).start()


@app.on_event("shutdown")
def on_shutdown():
    registry.stop_all()


@app.get("/api/status")
def status():
    return {
        "model_ready": vlm.ready,
        "model_error": vlm.error,
        "cameras_online": len(registry.list()),
    }


@app.get("/api/cameras")
def list_cameras():
    out = []
    for cam in CAMERAS:
        feed = registry.get(cam["id"])
        out.append(
            {
                **cam,
                "online": feed is not None and feed.get_jpeg() is not None,
            }
        )
    return out


@app.get("/api/prompts")
def suggested_prompts():
    return SUGGESTED_PROMPTS


def _mjpeg_generator(cam_id: str):
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
        time.sleep(1 / 25)


@app.get("/api/cameras/{cam_id}/stream")
def camera_stream(cam_id: str):
    if registry.get(cam_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras/{cam_id}/snapshot")
def camera_snapshot(cam_id: str):
    feed = registry.get(cam_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    jpeg = feed.get_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Camera has no frame yet")
    from fastapi.responses import Response

    return Response(content=jpeg, media_type="image/jpeg")


class ChatRequest(BaseModel):
    camera_id: str
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not vlm.ready:
        raise HTTPException(
            status_code=503,
            detail=vlm.error or "Model is still loading, please wait a moment.",
        )

    feed = registry.get(req.camera_id)
    if feed is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    frame = feed.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera has no frame yet")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    try:
        answer = vlm.ask(image, req.question)
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"camera_id": req.camera_id, "question": req.question, "answer": answer}


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
