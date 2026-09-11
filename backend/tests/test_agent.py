from types import SimpleNamespace
from unittest.mock import Mock

from app.agent import answer_health_question, wants_health_tool


def test_health_tool_intent_does_not_capture_visual_question():
    assert wants_health_tool("Which cameras are offline?")
    assert wants_health_tool("Is the Lobby camera working?")
    assert not wants_health_tool("What is the person carrying?")


def test_health_answer_uses_runtime_state_without_vlm():
    camera = SimpleNamespace(
        id="cam1", name="Lobby", location="Entrance", zone_tags=["door"], source_type="rtsp",
    )
    feed = SimpleNamespace(is_online=lambda: True, get_fps=lambda: 19.5)
    registry = SimpleNamespace(get=lambda camera_id: feed)
    cpu_monitor = SimpleNamespace(status=lambda camera_id: {"online": True})
    db = Mock()
    db.get.return_value = SimpleNamespace(settings={"enabled": True})
    vlm = SimpleNamespace(ready=False)

    result = answer_health_question(db, "Is Lobby working?", [camera], registry, cpu_monitor, vlm)
    assert "online at 19.5 FPS" in result["answer"]
    assert result["plan"] == {"tool": "camera_health", "camera_ids": ["cam1"]}
    assert result["details"][0]["cpu_online"] is True
