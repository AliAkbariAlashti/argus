import json
import stat

import httpx
import pytest
from PIL import Image

from app.runtime import VisionRuntime


def settings(**overrides):
    return {"provider": "compatible", "base_url": "http://localhost:8080/v1", "model": "test-vision", "api_key": "test-secret", "allow_remote": False, "monitor_enabled": False, **overrides}


@pytest.fixture
def runtime(tmp_path):
    return VisionRuntime(tmp_path / "runtime.json")


def mock_server(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr("app.runtime.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_no_ai_requires_no_model_load(runtime, monkeypatch):
    monkeypatch.setattr(runtime._embedded, "load", lambda: pytest.fail("Should not load Qwen"))
    runtime.load()
    assert not runtime.ready
    with pytest.raises(RuntimeError, match="Set up"):
        runtime.ask(Image.new("RGB", (16, 16)), "What do you see?")


def test_settings_permissions_secret_redaction_and_host_change(runtime):
    runtime.configure(settings())
    assert "api_key" not in runtime.public_settings()
    assert runtime.public_settings()["has_api_key"]
    assert stat.S_IMODE(runtime.path.stat().st_mode) == 0o600
    runtime.configure(settings(api_key=None))
    assert runtime._settings["api_key"] == "test-secret"
    runtime.configure(settings(base_url="http://127.0.0.1:8080/v1", api_key=None))
    assert runtime._settings["api_key"] == ""


def test_external_server_requires_explicit_consent(runtime):
    with pytest.raises(ValueError, match="Confirm"):
        runtime.configure(settings(base_url="https://example.test/v1"))
    with pytest.raises(ValueError, match="without credentials"):
        runtime.configure(settings(base_url="http://key:secret@localhost/v1"))


def test_vision_probe_and_stream_preserve_images_and_tokens(runtime, monkeypatch):
    requests = []
    def handler(req):
        payload = json.loads(req.content)
        requests.append(payload)
        assert req.headers["Authorization"] == "Bearer test-secret"
        assert payload["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/")
        if not payload["stream"]:
            return httpx.Response(200, json={"choices": [{"message": {"content": "The image is red."}}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"A person "}}]}\n\ndata: {"choices":[{"delta":{"content":"is visible."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    mock_server(monkeypatch, handler)
    runtime.configure(settings())
    assert not runtime.ready
    assert runtime.test()["ready"]
    tokens = []
    answer = runtime.ask([Image.new("RGB", (16, 16))] * 2, "Describe", fps=1, on_token=tokens.append)
    assert answer == "A person is visible."
    assert tokens == ["A person ", "is visible."]
    assert "chronological" in requests[-1]["messages"][-1]["content"][-1]["text"]
    runtime.configure(settings())
    assert not runtime.ready


def test_ollama_uses_native_cpu_api_and_container_host_alias(runtime, monkeypatch):
    requests = []

    def handler(req):
        payload = json.loads(req.content)
        requests.append((req.url.path, payload))
        assert req.url.path == "/api/generate"
        assert payload["options"]["num_gpu"] == 0
        assert payload["images"]
        assert len(payload["images"]) == 1
        assert not payload["images"][0].startswith("data:")
        if not payload["stream"]:
            return httpx.Response(200, json={"response": "The image is red.", "done": True})
        return httpx.Response(200, text='{"response":"red ","done":false}\n{"response":"image","done":true}\n')

    mock_server(monkeypatch, handler)
    monkeypatch.setattr("app.runtime.os.path.exists", lambda path: path == "/.dockerenv")
    monkeypatch.setenv("ARGUS_OLLAMA_RELAY_PORT", "11435")
    runtime.configure(settings(base_url="http://localhost:11434/v1", model="moondream:latest", api_key=""))
    assert runtime._request_url(runtime._settings["base_url"]) == "http://host.docker.internal:11435/v1"
    assert runtime.test()["ready"]
    tokens = []
    assert runtime.ask(Image.new("RGB", (16, 16)), "Describe", on_token=tokens.append) == "red image"
    assert tokens == ["red ", "image"]
    assert len(requests) == 2
    assert runtime.ask([Image.new("RGB", (16, 16))] * 2, "Describe") == "The image is red."
    assert len(requests) == 3


def test_disconnect_is_not_a_complete_answer(runtime, monkeypatch):
    mock_server(monkeypatch, lambda req: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n'))
    runtime.configure(settings())
    runtime._verified = True
    with pytest.raises(RuntimeError, match="disconnected"):
        runtime.ask(Image.new("RGB", (16, 16)), "Describe", on_token=lambda token: None)


def test_failed_probe_cannot_enable_chat(runtime, monkeypatch):
    mock_server(monkeypatch, lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "I cannot view images"}}]}))
    runtime.configure(settings())
    with pytest.raises(ValueError, match="did not identify"):
        runtime.test()
    assert not runtime.ready


def test_error_does_not_expose_provider_body_or_key(runtime, monkeypatch):
    mock_server(monkeypatch, lambda req: httpx.Response(401, text="secret provider internals"))
    runtime.configure(settings())
    with pytest.raises(RuntimeError) as error:
        runtime.test()
    assert "401" in str(error.value)
    assert "secret" not in str(error.value)


def test_provider_failure_disables_ai_without_switching_provider(runtime, monkeypatch):
    mock_server(monkeypatch, lambda req: httpx.Response(503))
    runtime.configure(settings())
    runtime._verified = True
    with pytest.raises(RuntimeError):
        runtime.ask(Image.new("RGB", (16, 16)), "Describe")
    assert not runtime.ready
    assert runtime.public_settings()["provider"] == "compatible"
    assert "503" in runtime.error


def test_saved_profiles_can_be_switched_without_exposing_keys(runtime):
    runtime.configure({**settings(model="vision-a"), "name": "Office model"})
    first_id = runtime.public_settings()["profile_id"]
    runtime.configure({**settings(model="vision-b", api_key="secret-b"), "name": "Backup model", "create_new": True, "activate": False})
    profiles = runtime.public_profiles()
    assert len(profiles) == 2
    assert {profile["name"] for profile in profiles} == {"Office model", "Backup model"}
    assert all("api_key" not in profile for profile in profiles)
    runtime.activate(next(profile["id"] for profile in profiles if profile["name"] == "Backup model"))
    assert runtime.public_settings()["model"] == "vision-b"
    reloaded = VisionRuntime(runtime.path)
    assert reloaded.public_settings()["model"] == "vision-b"
    assert reloaded.public_settings()["profile_id"] != first_id
