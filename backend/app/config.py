import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", BASE_DIR / "videos"))

MODEL_ID = os.environ.get("QWENVL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

# Seconds between frame samples used for live "current state" snapshots per camera.
FRAME_SAMPLE_INTERVAL = float(os.environ.get("FRAME_SAMPLE_INTERVAL", "1.0"))

# JPEG quality for MJPEG stream / snapshots served to the frontend.
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

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
]

SUGGESTED_PROMPTS = [
    "Describe this image in detail.",
    "What is happening in this image?",
    "What objects are visible in the image?",
    "How many people are visible?",
    "What is the person doing?",
    "What is the person holding?",
    "What kind of environment is shown?",
    "Is there anything unusual or potentially dangerous in this image?",
    "Read and extract all visible text from this image.",
    "Return a JSON description with people, objects, actions, and environment.",
]
