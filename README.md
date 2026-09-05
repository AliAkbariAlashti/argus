# Sentinel Vision

AI vision chatbot for CCTV monitoring — MVP. Point a Qwen2.5-VL-7B vision-language
model at live camera feeds and ask natural-language questions about what's
currently happening on screen.

For this MVP, "cameras" are looping local video files that simulate live feeds.
Swapping in real RTSP/webcam sources later only requires changing how
`CameraFeed` opens its source in [backend/app/camera.py](backend/app/camera.py) —
everything downstream (streaming, snapshots, chat) is unchanged.

## Architecture

```
videos/*.mp4  →  CameraFeed (loops file, decodes latest frame)  →  MJPEG stream (viewer)
                                                                 →  snapshot (VLM input)
                        Qwen2.5-VL-7B-Instruct  ←  /api/chat  ←  frontend chat panel
```

- `backend/` — FastAPI app: camera loop manager, model wrapper, REST API, serves the frontend.
- `frontend/` — static HTML/CSS/JS single-page app. No build step.
- `videos/` — sample footage used as simulated camera feeds.

## Running on the GPU VM

```bash
cd ~/dok
git clone <this-repo-url> qwenvl
cd qwenvl
source ~/dok/venv/bin/activate   # existing venv already has torch/transformers/etc.
bash backend/run.sh              # starts on :8000
```

Then open `http://<vm-ip>:8000` in a browser.

First request after startup will be slow while Qwen2.5-VL-7B loads into GPU
memory (~15-20s). Check `/api/status` to see when `model_ready` is `true`.

### Config

Environment variables (all optional, see [backend/app/config.py](backend/app/config.py)):

- `VIDEOS_DIR` — path to video files (default: `<repo>/videos`)
- `QWENVL_MODEL_ID` — HF model id (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
- `PORT` — server port (default: `8000`)
