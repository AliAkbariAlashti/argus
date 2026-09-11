import json
from types import SimpleNamespace

import pytest

from app.agent_runtime import AgentRuntime, AgentUnavailable


def test_unconfigured_agent_is_unavailable():
    runtime = AgentRuntime(base_url="", model="")
    assert runtime.public_status()["configured"] is False
    with pytest.raises(AgentUnavailable, match="is configured"):
        runtime._complete([])


def test_agent_runs_multiple_tools_then_answers(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=4)
    replies = iter([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "list_cameras", "arguments": "{}"}}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {"name": "get_camera_health", "arguments": json.dumps({"camera_ids": ["cam-1"]})}}]},
        {"role": "assistant", "content": "Camera One is online."},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages: next(replies))
    calls = []

    def fake_tool(name, args, db, registry, cpu_monitor, vlm):
        calls.append((name, args))
        return ({"ok": True}, ["cam-1"], None)

    monkeypatch.setattr(runtime, "_run_tool", fake_tool)
    statuses = []
    result = runtime.answer("Is Camera One online?", [], object(), object(), object(), object(), statuses.append)
    assert result["answer"] == "Camera One is online."
    assert result["cameras_used"] == ["cam-1"]
    assert [name for name, _ in calls] == ["list_cameras", "get_camera_health"]
    assert len(result["tool_trace"]) == 2
    assert statuses == ["Using list cameras…", "Using get camera health…"]


def test_agent_stops_at_step_limit(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=2)
    monkeypatch.setattr(runtime, "_complete", lambda messages: {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "unknown", "arguments": "{}"}}]})
    monkeypatch.setattr(runtime, "_run_tool", lambda *args: ({"error": "unknown"}, [], None))
    with pytest.raises(AgentUnavailable, match="2-step limit"):
        runtime.answer("keep going", [], object(), object(), object(), object())


def test_agent_accepts_invalid_tool_arguments_without_crashing(monkeypatch):
    runtime = AgentRuntime(base_url="http://model", model="instruct", max_steps=2)
    replies = iter([
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "unknown", "arguments": "not-json"}}]},
        {"role": "assistant", "content": "I could not run that tool."},
    ])
    monkeypatch.setattr(runtime, "_complete", lambda messages: next(replies))
    seen = []
    monkeypatch.setattr(runtime, "_run_tool", lambda name, args, *rest: (seen.append(args) or {"error": "unknown"}, [], None))
    result = runtime.answer("test", [], object(), object(), object(), SimpleNamespace())
    assert seen == [{}]
    assert result["answer"] == "I could not run that tool."
