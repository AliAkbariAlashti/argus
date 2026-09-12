import json
from types import SimpleNamespace

import pytest

from app.agent_runtime import AgentRuntime, AgentUnavailable


def fake_db():
    return SimpleNamespace(add=lambda row: None, commit=lambda: None)


def test_unconfigured_agent_is_unavailable():
    runtime = AgentRuntime(base_url="", model="")
    assert runtime.public_status()["configured"] is False
    with pytest.raises(AgentUnavailable, match="is configured"):
        runtime._complete([])


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
