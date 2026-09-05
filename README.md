# Sentinel Vision

AI vision chatbot for CCTV monitoring — MVP. Point a Qwen2.5-VL-7B vision-language
model at live camera feeds and ask natural-language questions about what's
currently happening on screen.

For this MVP, "cameras" are looping local video files that simulate live feeds.
Adding a real IP camera (RTSP) is exposed in the UI as a coming-soon option;
swapping it in later only requires changing how `CameraFeed` opens its source
in [backend/app/camera.py](backend/app/camera.py) — everything downstream
(streaming, snapshots, chat) is unchanged.

## Architecture

```
videos/*.mp4  →  CameraFeed (loops file, decodes latest frame)  →  MJPEG stream (viewer)
                                                                 →  snapshot (VLM input)
Postgres (camera metadata: name, location, zone tags, description)
                        Qwen2.5-VL-7B-Instruct  ←  /api/chat  ←  frontend chat panel
```

A question like "what's in the parking lot" is grounded before it ever reaches
the model: the camera's own metadata (name/location/zone tags/description) is
folded into the prompt so the model answers in context, instead of the app
running a separate retrieval step. See [backend/app/grounding.py](backend/app/grounding.py).

- `backend/` — FastAPI app: camera loop manager, model wrapper, camera CRUD, REST API, serves the frontend.
- `frontend/` — static HTML/CSS/JS single-page app (Dashboard / Chat / Alerts* / Archive* / Cameras). No build step.
- `videos/` — sample + uploaded footage used as simulated camera feeds.
- `docker-compose.yml` — Postgres for camera metadata.

## Running on the GPU VM

```bash
cd ~/dok
git clone <this-repo-url> qwenvl
cd qwenvl

docker compose up -d          # starts Postgres on :5432

source ~/dok/venv/bin/activate   # existing venv already has torch/transformers/etc.
pip install sqlalchemy "psycopg[binary]"   # new deps for camera CRUD storage

bash backend/run.sh              # starts on :8000
```

Then open `http://<vm-ip>:8000` in a browser.

First request after startup will be slow while Qwen2.5-VL-7B loads into GPU
memory (~15-20s). Check `/api/status` to see when `model_ready` is `true`.
On first boot, the cameras table is empty and gets seeded automatically with
the 4 sample videos in `videos/`.

### Config

Environment variables (all optional, see [backend/app/config.py](backend/app/config.py)):

- `DATABASE_URL` — Postgres connection string (default matches `docker-compose.yml`: `postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinelvision`)
- `VIDEOS_DIR` — path to video files (default: `<repo>/videos`)
- `QWENVL_MODEL_ID` — HF model id (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
- `PORT` — server port (default: `8000`)
