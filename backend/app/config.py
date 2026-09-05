import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", BASE_DIR / "videos"))

MODEL_ID = os.environ.get("QWENVL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

# Seconds between frame samples used for live "current state" snapshots per camera.
FRAME_SAMPLE_INTERVAL = float(os.environ.get("FRAME_SAMPLE_INTERVAL", "1.0"))

# JPEG quality for MJPEG stream / snapshots served to the frontend.
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# Cameras: id -> (display name, location label, video filename in VIDEOS_DIR)
CAMERAS = [
    {
        "id": "cam-01",
        "name": "Server Room",
        "location": "Building A - Basement",
        "file": "mixkit-thief-in-server-room-being-recorded-by-a-camera-23498-hd-ready.mp4",
    },
    {
        "id": "cam-02",
        "name": "East Corridor",
        "location": "Building A - Floor 2",
        "file": "mixkit-scientist-walking-down-a-corridor-4747-hd-ready.mp4",
    },
    {
        "id": "cam-03",
        "name": "Front Street",
        "location": "Main Entrance - Exterior",
        "file": "mixkit-street-with-people-walking-at-dusk-3428-hd-ready.mp4",
    },
    {
        "id": "cam-04",
        "name": "Reception Desk",
        "location": "Building A - Lobby",
        "file": "mixkit-hands-of-a-person-typing-on-a-cell-phone-4915-hd-ready.mp4",
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
