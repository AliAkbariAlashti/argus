import json
from types import SimpleNamespace

import pytest
import httpx

from app.agent_runtime import AgentRuntime, AgentUnavailable, CALL_ARGUS_TOOL, TOOL_CATALOG


def fake_db():
    return SimpleNamespace(add=lambda row: None, commit=lambda: None)


class FakeCameraQuery:
    def __init__(self, rows): self.rows = rows
    def order_by(self, *_args): return self
    def all(self): return self.rows


class FakeCameraDb:
    def __init__(self, rows): self.rows = rows
    def query(self, *_args): return FakeCameraQuery(self.rows)


def test_unconfigured_agent_is_unavailable():
    runtime = AgentRuntime(base_url="", model="")
    assert runtime.public_status()["configured"] is False
    with pytest.raises(AgentUnavailable, match="is configured"):
        runtime._complete([])


def test_compact_catalog_exposes_required_and_optional_arguments():
    assert "inspect_recording(camera_id,offset_seconds,question)" in TOOL_CATALOG
    assert "get_camera_health(camera_ids?)" in TOOL_CATALOG
    assert "Get live stream and CPU-analysis health" in TOOL_CATALOG


def test_agent_runs_multiple_tools_then_answers(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=5)
    replies = iter([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "load", "type": "function", "function": {"name": "load_tools", "arguments": json.dumps({"names": ["list_cameras", "get_camera_health"]})}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "list_cameras", "arguments": "{}"}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {"name": "get_camera_health", "arguments": json.dumps({"camera_ids": ["cam-1"]})}}]},
        {"role": "assistant", "content": "Camera One is online."},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages, tools=None, on_token=None: next(replies))
    calls = []

    def fake_tool(name, args, db, registry, cpu_monitor, vlm, session_id):
        calls.append((name, args))
        return ({"ok": True}, ["cam-1"], None)

    monkeypatch.setattr(runtime, "_run_tool", fake_tool)
    statuses = []
    result = runtime.answer("Is Camera One online?", [], fake_db(), object(), object(), object(), statuses.append)
    assert result["answer"] == "Camera One is online."
    assert result["cameras_used"] == ["cam-1"]
    assert [name for name, _ in calls] == ["list_cameras", "get_camera_health"]
    assert len(result["tool_trace"]) == 3
    assert statuses == ["Using load tools…", "Using list cameras…", "Using get camera health…"]


def test_agent_stops_at_step_limit(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=2)
    monkeypatch.setattr(runtime, "_complete", lambda messages, tools=None, on_token=None: {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "unknown", "arguments": "{}"}}]})
    monkeypatch.setattr(runtime, "_run_tool", lambda *args: ({"error": "unknown"}, [], None))
    with pytest.raises(AgentUnavailable, match="2-step limit"):
        runtime.answer("keep going", [], fake_db(), object(), object(), object())


def test_agent_accepts_invalid_tool_arguments_without_crashing(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=2)
    replies = iter([
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "unknown", "arguments": "not-json"}}]},
        {"role": "assistant", "content": "I could not run that tool."},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages, tools=None, on_token=None: next(replies))
    seen = []
    monkeypatch.setattr(runtime, "_run_tool", lambda name, args, *rest: (seen.append(args) or {"error": "unknown"}, [], None))
    result = runtime.answer("test", [], fake_db(), object(), object(), SimpleNamespace())
    assert seen == []
    assert result["tool_trace"][0]["result"]["error"].startswith("Tool is not loaded")
    assert result["answer"] == "I could not run that tool."


def test_structured_context_keeps_stable_follow_up_ids():
    context = AgentRuntime._remember({}, "search_observations", {"observations": [
        {"id": "obs-1", "camera_id": "cam-1", "observed_at": "2026-01-01T00:00:00Z", "object_type": "person", "track_id": "track-7", "global_entity_id": None},
    ]}, ["cam-1"])
    assert context["camera_ids"] == ["cam-1"]
    assert context["observations"][0]["id"] == "obs-1"
    assert context["observations"][0]["track_id"] == "track-7"


def test_streamed_tool_arguments_are_assembled_by_index():
    calls = []
    AgentRuntime._merge_tool_delta(calls, {"index": 0, "id": "call-1", "function": {"name": "get_camera_", "arguments": '{"camera'}})
    AgentRuntime._merge_tool_delta(calls, {"index": 0, "function": {"name": "health", "arguments": '_ids":["cam-1"]}'}})
    assert calls == [{"id": "call-1", "type": "function", "function": {"name": "get_camera_health", "arguments": '{"camera_ids":["cam-1"]}'}}]


def test_ollama_follow_up_messages_use_native_tool_format():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "list_cameras", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "name": "list_cameras", "content": '{"cameras":[]}'},
    ]
    assert AgentRuntime._ollama_messages(messages) == [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list_cameras", "arguments": {}}}]},
        {"role": "tool", "content": '{"cameras":[]}', "tool_name": "list_cameras"},
    ]


def test_ollama_uses_single_step_generic_dispatch(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    replies = iter([
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "call_argus_tool", "arguments": json.dumps({"name": "get_camera_health", "arguments": {}})}}]},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages, tools=None, on_token=None: next(replies))
    calls = []
    monkeypatch.setattr(runtime, "_run_tool", lambda name, args, *rest: (calls.append((name, args)) or {"ok": True}, [], None))
    result = runtime.answer("Which cameras are online?", [], fake_db(), object(), object(), object())
    assert calls == [("get_camera_health", {})]
    assert result["answer"] == "0 of 0 cameras are online."


def test_ollama_can_chain_tools_before_answering(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen", max_steps=4)
    replies = iter([
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "call_argus_tool", "arguments": json.dumps({"name": "search_observations", "arguments": {"object_type": "person"}})}}]},
        {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "call_argus_tool", "arguments": json.dumps({"name": "inspect_live_cameras", "arguments": {"camera_ids": ["cam-1"], "question": "Is the person still there?"}})}}]},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages, tools=None, on_token=None: next(replies))
    calls = []
    def run_tool(name, args, *rest):
        calls.append((name, args))
        if name == "search_observations":
            return {"count": 1, "observations": [{"camera_id": "cam-1"}]}, ["cam-1"], "stored"
        return {"answer": "The person is still visible."}, ["cam-1"], "live"
    monkeypatch.setattr(runtime, "_run_tool", run_tool)
    result = runtime.answer("Find the recent person and check now.", [], fake_db(), object(), object(), object())
    assert [name for name, _ in calls] == ["search_observations", "inspect_live_cameras"]
    assert result["answer"] == "The person is still visible."
    assert result["snapshot"] == "stored"
    assert result["steps"] == 2


def test_agent_builds_compact_evidence_metadata():
    evidence = AgentRuntime._evidence_items("search_observations", {"observations": [{
        "id": "obs-1", "camera_id": "cam-1", "observed_at": "2026-09-14T12:00:00Z",
        "object_type": "person", "attributes": {"snapshot": "available"},
    }]})
    assert evidence == [{"kind": "observation", "observation_id": "obs-1", "camera_id": "cam-1",
                         "timestamp": "2026-09-14T12:00:00Z", "label": "person"}]
    assert "snapshot" not in str(evidence)


def test_ollama_constrained_plan_becomes_validated_dispatch(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    packets = iter([
        {"message": {"role": "assistant", "content": json.dumps({"tool": "get_camera_health", "answer": ""})}},
    ])
    seen = []
    def post(url, **kwargs):
        seen.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: next(packets))
    monkeypatch.setattr("app.agent_runtime.httpx.post", post)
    message = runtime._complete([{"role": "user", "content": "health"}], [CALL_ARGUS_TOOL])
    arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"name": "get_camera_health", "arguments": {}}
    assert "tools" not in seen[0]
    assert seen[0]["format"]["properties"]["tool"]["enum"]
    assert len(seen) == 1


def test_ollama_argument_extraction_drops_ungrounded_camera_ids(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    monkeypatch.setattr("app.agent_runtime.httpx.post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"message": {"content": json.dumps({"camera_ids": ["camera1", "camera2"]})}},
    ))
    result = runtime._ollama_extract_arguments(
        [{"role": "user", "content": "Give me an exact activity report for the last 24 hours."}],
        "get_activity_report",
    )
    assert result == {}


def test_ollama_argument_extraction_drops_all_cameras_scope(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    monkeypatch.setattr("app.agent_runtime.httpx.post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"message": {"content": json.dumps({"camera_id": "all cameras", "target": "person carrying a backpack"})}},
    ))
    result = runtime._ollama_extract_arguments(
        [{"role": "user", "content": "Create a rule for a person carrying a backpack on all cameras."}],
        "propose_alert_rule",
    )
    assert result == {"target": "person carrying a backpack"}


def test_ollama_argument_extraction_requires_camera_ids_from_directory(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    monkeypatch.setattr("app.agent_runtime.httpx.post", lambda *args, **kwargs: pytest.fail("unexpected model call"))
    result = runtime._ollama_extract_arguments(
        [
            {"role": "system", "content": 'Camera directory:\n[{"id":"camera-123","name":"Lobby"}]'},
            {"role": "user", "content": "Check camera-123."},
        ],
        "get_camera_health",
    )
    assert result == {"camera_ids": ["camera-123"]}


def test_camera_only_arguments_bind_without_a_second_model_call(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    monkeypatch.setattr("app.agent_runtime.httpx.post", lambda *args, **kwargs: pytest.fail("unexpected model call"))
    messages = [
        {"role": "system", "content": 'Camera directory. Resolve camera names to these exact stable IDs:\n[{"id":"cam-1","name":"Server Room"}]'},
        {"role": "user", "content": "Is the Server Room camera online?"},
    ]
    assert runtime._ollama_extract_arguments(messages, "get_camera_health") == {"camera_ids": ["cam-1"]}


def test_camera_only_arguments_default_to_all_without_a_second_model_call(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    monkeypatch.setattr("app.agent_runtime.httpx.post", lambda *args, **kwargs: pytest.fail("unexpected model call"))
    assert runtime._ollama_extract_arguments(
        [{"role": "user", "content": "Which cameras are online?"}], "get_camera_health"
    ) == {}


def test_camera_health_is_compact_for_model_context():
    result = {"cameras": [{"camera_id": "cam-1", "name": "Lobby", "online": True, "fps": 20,
                            "cpu_enabled": True, "cpu_status": {"online": True, "motion": False,
                            "boxes": [[0, 0, 1, 1]], "tracked_objects": [{"track_id": 1}],
                            "tracked_counts": {"person": 1}}}]}
    compact = AgentRuntime._model_result("get_camera_health", result)
    assert compact["cameras"] == [["cam-1", "Lobby", True, 20, True, [], {"person": 1}, None]]


def test_exact_operational_result_does_not_need_second_model_call(monkeypatch):
    runtime = AgentRuntime(base_url="http://localhost:11434/v1", model="qwen")
    reply = {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "call_argus_tool", "arguments": json.dumps({"name": "list_cameras", "arguments": {}})}}]}
    calls = []
    monkeypatch.setattr(runtime, "_complete", lambda *args, **kwargs: calls.append(1) or reply)
    monkeypatch.setattr(runtime, "_run_tool", lambda *args: ({"cameras": [{"name": "Lobby", "location": "Main", "online": True}]}, ["cam-1"], None))
    result = runtime.answer("list", [], fake_db(), object(), object(), object())
    assert calls == [1]
    assert "Lobby (Main) — online" in result["answer"]


def test_required_tool_arguments_are_rejected_before_execution():
    runtime = AgentRuntime(base_url="", model="")
    result, used, evidence = runtime._run_tool("propose_alert_rule", {}, FakeCameraDb([]), object(), object(), object())
    assert result == {"error": "Missing required tool arguments: target."}
    assert used == [] and evidence is None


def test_camera_names_resolve_to_stable_ids_before_tool_execution():
    cameras = [SimpleNamespace(id="cam-1", name="Server Room")]
    args, error = AgentRuntime._resolve_camera_arguments(
        {"camera_ids": ["Server Room"], "camera_id": "cam-1"}, cameras
    )
    assert error is None
    assert args == {"camera_ids": ["cam-1"], "camera_id": "cam-1"}


def test_unknown_and_ambiguous_camera_names_are_rejected():
    cameras = [SimpleNamespace(id="cam-1", name="Lobby"), SimpleNamespace(id="cam-2", name="Lobby")]
    _, ambiguous = AgentRuntime._resolve_camera_arguments({"camera_id": "Lobby"}, cameras)
    _, unknown = AgentRuntime._resolve_camera_arguments({"camera_ids": ["Roof"]}, cameras)
    assert "ambiguous" in ambiguous
    assert "Roof" in unknown


def test_list_recordings_reports_duration_without_exposing_path(tmp_path, monkeypatch):
    recording = tmp_path / "camera.mp4"
    recording.touch()
    camera = SimpleNamespace(id="cam-1", name="Lobby", source_type="file", source_path=recording.name, created_at=None)
    capture = SimpleNamespace(get=lambda prop: 20 if prop == 5 else 200, release=lambda: None)
    monkeypatch.setattr("app.agent_runtime.VIDEOS_DIR", tmp_path)
    monkeypatch.setattr("app.agent_runtime.cv2.VideoCapture", lambda _path: capture)
    runtime = AgentRuntime(base_url="", model="")
    result, used, _ = runtime._run_tool("list_recordings", {}, FakeCameraDb([camera]), object(), object(), object())
    assert result == {"recordings": [{"camera_id": "cam-1", "camera_name": "Lobby", "available": True, "duration_seconds": 10.0, "fps": 20.0}]}
    assert used == ["cam-1"]
    assert str(recording) not in str(result)


def test_failed_endpoint_enters_fast_fallback_cooldown(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct")
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise httpx.ConnectError("down")
    monkeypatch.setattr("app.agent_runtime.httpx.post", fail)
    with pytest.raises(AgentUnavailable, match="down"):
        runtime._complete([])
    with pytest.raises(AgentUnavailable, match="down"):
        runtime._complete([])
    assert len(calls) == 1
    assert runtime.public_status()["cooling_down"] is True


def test_agent_configuration_persists_without_exposing_key(tmp_path):
    path = tmp_path / "agent.json"
    runtime = AgentRuntime(base_url="", model="", path=path)
    status = runtime.configure(base_url="http://127.0.0.1:11434/v1", model="qwen", api_key="secret", max_steps=6)
    assert status["has_api_key"] is True
    assert "secret" not in str(status)
    reloaded = AgentRuntime(base_url="", model="", path=path)
    assert reloaded.model == "qwen"
    assert reloaded.api_key == "secret"
    assert reloaded.max_steps == 6
    assert reloaded.public_status()["ready"] is False
