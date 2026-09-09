"""Integration checks against an explicitly isolated PostgreSQL test database."""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("ARGUS_TEST_DATABASE"), reason="Set ARGUS_TEST_DATABASE and DATABASE_URL to an isolated test DB")


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main
    from app.runtime import VisionRuntime
    runtime = VisionRuntime(tmp_path / "runtime.json")
    monkeypatch.setattr(main, "vlm", runtime)
    monkeypatch.setattr(main.monitor, "start", lambda: None)
    with TestClient(main.app) as client:
        client.delete("/api/chat/history")
        yield client, main, runtime


def test_no_ai_keeps_camera_ui_and_metadata_available(client):
    c, main, _ = client
    assert c.get("/api/status").json()["configured"] is False
    cameras = c.get("/api/cameras").json()
    assert cameras
    response = c.post("/api/chat/stream", json={"question": "What is happening?", "scope": "camera", "focused_camera_id": cameras[0]["id"]})
    assert "event: error" in response.text
    assert "AI setup" in response.text
    assert c.get("/api/chat/history").json() == []
    response = c.post("/api/chat/stream", json={"question": "How many cameras are there?"})
    assert "event: answer" in response.text
    assert "event: done" in response.text
    assert not main._chat_lock.locked()


def test_stream_saves_answer_and_evidence(client, monkeypatch):
    from PIL import Image
    c, main, runtime = client
    runtime._settings["provider"] = "compatible"
    runtime._verified = True
    def answer(images, question, **kwargs):
        kwargs["on_token"]("Visible ")
        kwargs["on_token"]("scene.")
        return "Visible scene."
    monkeypatch.setattr(runtime, "ask", answer)
    monkeypatch.setattr(main, "_frame_sequence_to_images", lambda *args: [Image.new("RGB", (32, 32))])
    camera = c.get("/api/cameras").json()[0]
    response = c.post("/api/chat/stream", json={"question": "Describe the scene", "scope": "camera", "focused_camera_id": camera["id"]})
    assert response.text.count("event: token") == 2
    assert "event: answer" in response.text
    assert response.headers["x-accel-buffering"] == "no"
    history = c.get("/api/chat/history").json()
    assert len(history) == 2
    assert history[-1]["snapshot"].startswith("data:image/")
    assert history[-1]["cameras_used"] == [camera["id"]]


def test_busy_and_validation(client):
    c, main, _ = client
    main._chat_lock.acquire()
    try:
        assert c.post("/api/chat/stream", json={"question": "Describe"}).status_code == 429
        assert c.delete("/api/chat/history").status_code == 409
    finally:
        main._chat_lock.release()
    assert c.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert c.post("/api/chat/stream", json={"question": "x", "scope": "bogus"}).status_code == 422
