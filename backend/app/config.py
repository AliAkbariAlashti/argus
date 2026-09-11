import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", BASE_DIR / "videos"))

MODEL_ID = os.environ.get("QWENVL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

# Seconds between frame samples used for live "current state" snapshots per camera.
FRAME_SAMPLE_INTERVAL = float(os.environ.get("FRAME_SAMPLE_INTERVAL", "1.0"))

# JPEG quality for MJPEG stream / snapshots served to the frontend.
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# Recent-frame buffer used to give the model a sense of motion. Three frames
# a second apart is enough to tell "walking toward the door" from "standing
# still" without tripling VRAM use per request.
MOTION_BUFFER_SIZE = int(os.environ.get("MOTION_BUFFER_SIZE", "3"))
MOTION_BUFFER_SECONDS = float(os.environ.get("MOTION_BUFFER_SECONDS", "1.0"))

# Frames sent per camera on a single-camera question. Fleet-wide questions
# always use one frame per camera to keep the image count manageable.
CHAT_FRAMES_SINGLE_CAMERA = int(os.environ.get("CHAT_FRAMES_SINGLE_CAMERA", "3"))

# Vision-token budget per frame, in pixels — passed straight to Qwen2.5-VL's
# processor (its documented lever for resolution/VRAM tradeoff; the model
# tokenizes images at a resolution-dependent token count, so this is a much
# more direct control than resizing to a fixed edge length). The 7B model
# idles at ~18.5GB/20GB just loaded, so keep max_pixels conservative — a
# fleet question attaches one frame per camera in a single request.
VLM_MIN_PIXELS = int(os.environ.get("VLM_MIN_PIXELS", str(256 * 28 * 28)))
VLM_MAX_PIXELS = int(os.environ.get("VLM_MAX_PIXELS", str(768 * 28 * 28)))

# How many prior turns of the conversation to replay to the model.
CHAT_HISTORY_TURNS = int(os.environ.get("CHAT_HISTORY_TURNS", "6"))

# Run a tiny throwaway inference once the model loads, so the first real
# question doesn't pay the lazy-init cost mid-demo.
PREWARM_MODEL = os.environ.get("PREWARM_MODEL", "1") not in ("0", "false", "False")

# Seeded on first boot only (when the cameras table is empty), so the demo
# has content immediately. Afterwards cameras are fully managed via the
# CRUD API and persisted in Postgres.
SEED_CAMERAS = [
    {
        "name": "Server Room",
        "location": "Building A - Basement",
        "zone_tags": ["server-room", "data-center", "restricted"],
        "description": "Data center server room. Racks of networking and compute equipment. Restricted access area.",
        "source_type": "file",
        "source_path": "mixkit-thief-in-server-room-being-recorded-by-a-camera-23498-hd-ready.mp4",
    },
    {
        "name": "East Corridor",
        "location": "Building A - Floor 2",
        "zone_tags": ["corridor", "hallway", "indoor"],
        "description": "Indoor hallway on the second floor connecting offices and labs.",
        "source_type": "file",
        "source_path": "mixkit-scientist-walking-down-a-corridor-4747-hd-ready.mp4",
    },
    {
        "name": "Front Street",
        "location": "Main Entrance - Exterior",
        "zone_tags": ["street", "exterior", "entrance", "parking-lot"],
        "description": "Exterior view of the street and sidewalk in front of the main entrance.",
        "source_type": "file",
        "source_path": "mixkit-street-with-people-walking-at-dusk-3428-hd-ready.mp4",
    },
    {
        "name": "Reception Desk",
        "location": "Building A - Lobby",
        "zone_tags": ["lobby", "reception", "indoor", "entrance"],
        "description": "Front desk area in the main lobby, monitors visitor check-in.",
        "source_type": "file",
        "source_path": "mixkit-hands-of-a-person-typing-on-a-cell-phone-4915-hd-ready.mp4",
    },
    {
        "name": "Street Junction",
        "location": "Main Entrance - Street Level",
        "zone_tags": ["street", "exterior", "traffic", "crosswalk"],
        "description": "Overhead view of a busy street junction with pedestrian and vehicle traffic.",
        "source_type": "file",
        "source_path": "mixkit-crowds-of-people-cross-a-street-junction-4401-hd-ready.mp4",
    },
]

SUGGESTED_PROMPTS = [
    "What is happening right now?",
    "Describe the people and visible objects.",
    "What changed in the recent frames?",
    "Which cameras are online?",
]

# ---- Background event monitor ----
# Motion-gated, rate-limited VLM classification per camera, logged only on
# a state change (something entering/leaving view, a weapon appearing) —
# feeds the Alerts tab. Set MONITOR_ENABLED=0 to turn it off entirely if
# the extra VLM calls shouldn't compete with live chat for the one GPU.
MONITOR_ENABLED = os.environ.get("MONITOR_ENABLED", "1") not in ("0", "false", "False")

# How often to check each camera for motion (cheap, no model — plain frame
# diffing) before ever considering a VLM call.
MONITOR_POLL_SECONDS = float(os.environ.get("MONITOR_POLL_SECONDS", "3.0"))

# Fraction of downscaled pixels that must change between polls to count as
# motion. Tuned loosely against the sample looping clips — raise it if a
# camera with a static background triggers too often.
MONITOR_MOTION_THRESHOLD = float(os.environ.get("MONITOR_MOTION_THRESHOLD", "0.02"))

# Minimum time between VLM classification calls for the *same* camera, even
# under continuous motion. This shares one GPU (and its thin VRAM headroom)
# with live chat, so a busy camera can't be allowed to poll the model
# constantly.
MONITOR_MIN_VLM_INTERVAL_SECONDS = float(
    os.environ.get("MONITOR_MIN_VLM_INTERVAL_SECONDS", "20.0")
)

# Text instruction model used as the Argus agent's planner. This is separate
# from the vision model: the planner chooses tools and composes answers while
# the VLM is called only when a tool needs to inspect pixels.
AGENT_BASE_URL = os.environ.get("ARGUS_AGENT_BASE_URL", "").rstrip("/")
AGENT_MODEL = os.environ.get("ARGUS_AGENT_MODEL", "")
AGENT_API_KEY = os.environ.get("ARGUS_AGENT_API_KEY", "")
AGENT_MAX_STEPS = max(1, min(int(os.environ.get("ARGUS_AGENT_MAX_STEPS", "8")), 16))
AGENT_TIMEOUT_SECONDS = max(5, int(os.environ.get("ARGUS_AGENT_TIMEOUT_SECONDS", "90")))
