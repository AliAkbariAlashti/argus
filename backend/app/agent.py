"""Deterministic operational tools used before visual inference."""
import re

from .cpu import CpuSettings
from .models import CpuConfig

_HEALTH_RE = re.compile(
    r"\b(health|healthy|online|offline|connection|connected|disconnect|fps|frame rate|working|available)\b",
    re.IGNORECASE,
)


def wants_health_tool(question):
    return bool(_HEALTH_RE.search(question))


def _selected_cameras(question, cameras, forced_camera_id=None):
    if forced_camera_id:
        return [camera for camera in cameras if camera.id == forced_camera_id]
    lowered = question.lower()
    matched = [
        camera for camera in cameras
        if camera.name.lower() in lowered
        or (camera.location and camera.location.lower() in lowered)
        or any(tag.lower() in lowered for tag in (camera.zone_tags or []))
    ]
    return matched or cameras


def answer_health_question(db, question, cameras, registry, cpu_monitor, vlm, forced_camera_id=None):
    if not wants_health_tool(question):
        return None
    selected = _selected_cameras(question, cameras, forced_camera_id)
    lines, details = [], []
    for camera in selected:
        feed = registry.get(camera.id)
        online = bool(feed and feed.is_online())
        fps = feed.get_fps() if feed else 0.0
        config = db.get(CpuConfig, camera.id)
        cpu_enabled = CpuSettings(**(config.settings if config else {})).enabled
        cpu_state = cpu_monitor.status(camera.id)
        details.append({
            "camera_id": camera.id, "camera_name": camera.name, "source_type": camera.source_type,
            "online": online, "fps": fps, "cpu_enabled": cpu_enabled,
            "cpu_online": bool(cpu_state.get("online")), "cpu_message": cpu_state.get("message"),
        })
        state = f"online at {fps:g} FPS" if online else "offline"
        cpu = "CPU analysis active" if cpu_enabled and cpu_state.get("online") else "CPU analysis waiting" if cpu_enabled else "CPU analysis disabled"
        lines.append(f"{camera.name}: {state}; {cpu}.")
    offline = sum(not item["online"] for item in details)
    heading = f"Camera health: {len(details) - offline} of {len(details)} selected cameras online."
    model = "The configured AI model is ready." if vlm.ready else "The AI model is not ready; camera and CPU tools remain available."
    return {
        "answer": "\n\n".join([heading, *lines, model]),
        "cameras_used": [camera.id for camera in selected],
        "snapshot": None,
        "plan": {"tool": "camera_health", "camera_ids": [camera.id for camera in selected]},
        "details": details,
    }
