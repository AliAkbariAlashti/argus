# Argus — video intelligence MVP

Argus Agent is the primary interface and default landing page. Its conversation
routes operational questions through camera health, CPU analysis, structured
observations, historical evidence, selective VLM verification, and live camera
tools. The dashboard, activity, rules, cameras, health, CPU tools, and AI setup
remain available as focused controls in the sidebar.

The included sources are **looping demo video files**, clearly labeled in the UI.
Visual answers come from the configured model, never canned demo responses.
Live RTSP sources are supported. This version does not record RTSP footage or
include historical video retrieval.

## Start without a GPU

```bash
docker compose -f compose.mvp.yml up --build -d
```

Open **http://localhost:8090**. Five included videos populate a new database.
The CPU app can display sources, upload videos, manage rules, and show saved
activity. No model weights or CUDA packages are downloaded by this profile.
The database, uploaded media, and AI settings use separate persistent volumes.
Use `ARGUS_PORT=8092` if the default port is occupied.

### Connect a live RTSP camera

Open **Cameras**, choose **Add source**, select **IP Camera / RTSP**, and enter
the stream URL, for example `rtsp://192.168.1.50:8554/camera`. The Argus server
must be able to reach that address on the local network. Temporary disconnects
are retried automatically.

The MVP is a single trusted workspace without authentication or tenant isolation.
The Compose profile binds to localhost; use a local browser or SSH tunnel for a
private presentation. Add authentication and permissions before customer exposure.

### Connect the analyst

Open **AI setup** and choose:

| Connection | Requirements | Data path |
|---|---|---|
| Camera viewing only | CPU and PostgreSQL | No model calls |
| Local vision API | A running, image-capable model server | Frames go to your chosen server |
| Hosted vision API | Compatible endpoint, model name, API key if required | Sampled frames go to your provider; charges may apply |
| Embedded Qwen | GPU dependency profile and sufficient NVIDIA VRAM | Model inference on the Argus host |

Full setup guides (Ollama, llama.cpp, hosted APIs, embedded Qwen, environment
variables, troubleshooting) are served at `/docs/` once the app is running.

### CPU tools (no model required)

Every profile, including the plain CPU one, runs OpenCV measurements per camera
on a background thread: motion (frame differencing in a chosen watch area),
brightness/low-light, sharpness/blur, face detection (Haar cascade), and people
detection (HOG pedestrian descriptor). Face/people detection is heavier than the
other measurements, so it runs on a slower sampling cadence and only when
enabled per camera. None of this downloads model weights or calls a remote
service. See `/docs/#cpu` for details and limits.

For a compatible API, enter its base URL **including `/v1`**, exact vision model
name, and optional key. The protocol is streaming `/chat/completions` with image
inputs. Examples of compatible servers:
[Ollama](https://docs.ollama.com/api/openai-compatibility) and
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
Compatibility still depends on the specific model and server configuration.

Inside Docker, `localhost` refers to the app container, not your host. Use an
address reachable from the container for a model hosted elsewhere. Argus requires
explicit permission before sending frames to a non-loopback server, including LAN
servers. Keys are stored in a private settings file, excluded from Git and never
returned by the settings API. This is file protection, not encrypted secret storage.

**Save settings** does not contact the model. **Test vision connection** saves the
settings and sends one synthetic red image. A passing test enables visual chat.
Compatible APIs must be tested again after restarting the app. Embedded Qwen's
first test may download weights. The CPU Docker image does not contain its GPU
packages; run the GPU profile in a separately prepared environment.

Background visual monitoring is **off by default**. Enable it explicitly in setup;
it generates additional model requests. There is no automatic provider failover:
a disconnected model produces an actionable error, while camera viewing remains
available. Local CPU model serving is possible through compatible servers, but
latency and usable model size depend on that server's resources.

## Presentation flow

1. Open Overview: five source previews, actual recent detection counts, and AI status.
2. Choose a camera. Keep its video visible beside the analyst.
3. Ask “What is happening right now?” The model streams a single-camera answer.
4. Open the evidence snapshot to inspect the actual frame used.
5. Ask a follow-up, or select All cameras to compare current scenes. Fleet answers
   are delivered when complete so internal camera-routing markers stay hidden.
6. Open Activity to review existing detections and their snapshots. Enable monitoring
   and add a visible-condition rule to demonstrate sampled automatic observations.
7. Open AI setup to explain local, hosted, and GPU deployment options honestly.

Without a model connection, demonstrate viewing and setup; camera metadata questions
such as “How many cameras are there?” still work. Visual chat requires a real
vision backend. Camera status, event counts and analysis outputs are not fabricated.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-core.txt
# Point to a dedicated PostgreSQL database; do not reuse another application's DB.
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/argus'
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8090
```

`backend/requirements.txt` preserves the original GPU VM dependency profile and adds
HTTP support. It is not needed for CPU/API operation. `backend/run.sh` remains the
legacy GPU VM launcher. `docker-compose.yml` remains its original PostgreSQL setup;
use `compose.mvp.yml` for the new full app profile.

Settings default to `backend/data/runtime.json`. Override with `ARGUS_RUNTIME_FILE`.
Set `ARGUS_NODE_ID` to a stable deployment or edge-worker name when multiple
Argus processes write observations to the same database.

### Structured observation index

CPU object detections, tracked-object samples, line crossings, and dwell events
are stored as normalized observations in PostgreSQL. Query them through
`GET /api/observations` with optional `camera_id`, `object_type`, `kind`,
`action`, `track_id`, `producer_id`, `since`, and `until` filters. Observation
IDs and track instance IDs are generated by workers, so records from different
sites can be merged without database-sequence collisions.

Historical questions in Live analyst are planned against this index before any
vision-model call. Exact counts use unique track instances, camera/time filters
stay deterministic, and matching Activity evidence is returned when available.
Questions containing current-time language such as "right now" remain on the
live-frame path.

### Portable visual search

CPU object detections also create a versioned 512-dimensional appearance
descriptor for each object crop. The built-in `opencv-hsv` provider is a
lightweight colour and appearance matcher; it is not an identity or semantic
re-identification model. Vectors use ordinary JSON-compatible database columns,
so PostgreSQL vector extensions are optional.

`GET /api/visual-search/{observation_id}` finds similar indexed crops. Optional
`camera_id`, `object_type`, `since`, `until`, and `limit` filters narrow the
search. The portable backend evaluates at most
`ARGUS_VISUAL_SEARCH_MAX_CANDIDATES` rows (default `5000`) and reports when the
candidate set was truncated. Larger deployments can keep the same observation
contract while replacing the provider with CLIP/SigLIP and search with a native
vector index.

Set `ARGUS_VECTOR_BACKEND=pgvector` to use database-native cosine ranking. At
startup Argus tries to enable the PostgreSQL `vector` extension and add its
accelerated column. If the database image, hosted service, or database role
does not provide that extension, Argus reports the reason at
`GET /api/system/visual-search` and continues with bounded JSON search. Use a
pgvector-enabled PostgreSQL service and grant the application role permission
to create the extension, or have an administrator create it beforehand.

For semantic image/text search, install `backend/requirements-embeddings.txt`
and set `ARGUS_VISUAL_EMBEDDING_PROVIDER=clip`. The model is selected with
`ARGUS_SEMANTIC_MODEL` and defaults to `openai/clip-vit-base-patch32`; it may be
a downloaded model ID or a mounted local directory. New object crops then use
the same image/text vector space, and `GET /api/visual-search?q=red+vehicle`
searches them. Existing OpenCV vectors remain valid and searchable by reference;
changing providers does not reinterpret or delete earlier vectors.

### Selective VLM verification

Historical chat questions containing `verify`, `confirm`, `double-check`, or
`check the evidence` first run through the deterministic observation planner.
Argus selects distinct tracked objects with stored evidence, prefers coverage
across cameras, and sends at most `ARGUS_VLM_VERIFY_MAX_FRAMES` images to the
configured VLM (default `6`, hard maximum `12`). Responses include the exact
observation IDs, event IDs, camera names, and timestamps used as evidence.
Detector labels are retrieval hints; the VLM is instructed to report uncertainty
and never infer a shared identity across cameras.

### Cross-camera entity matching

Camera topology is configured as directed travel windows through
`GET/POST/DELETE /api/camera-links`. For example, Lobby → Hallway may allow
10–90 seconds while the reverse direction has a different route or no link.

`POST /api/entity-matches/suggest/{observation_id}` compares the tracked source
observation only with compatible embeddings from reachable cameras, within the
configured travel window and with the same detected object class. The minimum
similarity is controlled by `ARGUS_ENTITY_MATCH_MIN_SIMILARITY` (default `0.80`).
Suggestions remain reviewable records and do not imply identity.

Use `GET /api/entity-matches` to review suggestions and
`POST /api/entity-matches/{id}/decision` with `confirmed` or `rejected`.
Confirmation assigns one UUID `global_entity_id` to both complete tracks.
Conflicting previously confirmed identities are rejected instead of merged.

Other controls: `VIDEOS_DIR`, `QWENVL_MODEL_ID`, `CHAT_FRAMES_SINGLE_CAMERA`,
`VLM_MIN_PIXELS`, `VLM_MAX_PIXELS`, `CHAT_HISTORY_TURNS`, `MONITOR_ENABLED`,
`MONITOR_MIN_VLM_INTERVAL_SECONDS`, and `MONITOR_MOTION_THRESHOLD`.

## Validation

```bash
pip install pytest
PYTHONPATH=backend pytest backend/tests/test_runtime.py backend/tests/test_monitor.py backend/tests/test_camera.py
# Integration tests require an isolated PostgreSQL database. They clear chat history.
ARGUS_TEST_DATABASE=1 DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/argus_test' \
  PYTHONPATH=backend pytest backend/tests/test_chat_api.py
node --check frontend/app.js
# Browser checks against a running isolated app; AI responses are test fixtures only.
npm install
npx playwright install chromium
ARGUS_URL=http://127.0.0.1:8090 npm run test:ui
# Or use an installed browser: CHROME_PATH=/usr/bin/google-chrome
```

## Current limits

- One shared conversation and one active interactive request per server. Busy requests
  are rejected explicitly. No multi-user permissions yet.
- Analysis uses sampled frames; model observations are fallible. No calibrated
  accuracy claim, guaranteed detection latency, or production capacity claim.
- History contains conversations and event snapshots, not searchable recorded video.
- RTSP streams are viewed and analyzed live but are not recorded. Recording
  retention, incident acknowledgement and notification delivery are subsequent milestones.
- Configuration supports one active provider; automatic routing, cost quotas,
  heterogeneous device scheduling and production secrets management are future work.
# Argus vision agent

Argus uses two separate model roles. A text instruction model plans work and
calls camera tools; the configured vision model inspects selected frames. The
agent does not require the two roles to use the same provider or machine.

For a local Ollama planner, install a tool-capable instruction model and set:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ARGUS_AGENT_BASE_URL=http://host.docker.internal:11435/v1
ARGUS_AGENT_MODEL=qwen3:4b-instruct-2507-q4_K_M
```

`ARGUS_AGENT_BASE_URL` may instead point to any OpenAI-compatible endpoint.
When it is unset or unreachable, Argus keeps camera viewing and deterministic
queries available. `GET /api/agent/status` reports configuration and
`POST /api/agent/test` verifies the planner endpoint.
