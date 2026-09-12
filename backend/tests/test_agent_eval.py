from evals import run_agent_eval


def test_eval_requires_agent_scope_expected_tool_and_confirmation(monkeypatch):
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        if path == "/api/agent/status": return {"configured": True}
        if path == "/api/chat/sessions": return {"id": "eval-session"}
        if path == "/api/chat":
            return {"scope": "agent", "answer": "Prepared.", "tool_trace": [{"tool": "propose_alert_rule"}], "pending_actions": [{"id": "action-1"}]}
        return {"deleted": "eval-session"}

    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    result = run_agent_eval.run("http://argus", [{"name": "proposal", "question": "create rule", "expected_tools": ["propose_alert_rule"], "expects_confirmation": True, "max_seconds": 5}])
    assert result[0]["passed"] is True


def test_eval_reports_deterministic_fallback_as_failure(monkeypatch):
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        if path == "/api/agent/status": return {"configured": True}
        if path == "/api/chat/sessions": return {"id": "eval-session"}
        if path == "/api/chat": return {"scope": "camera_health", "answer": "fallback", "tool_trace": []}
        return {}

    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    result = run_agent_eval.run("http://argus", [{"name": "health", "question": "online?", "expected_tools": ["get_camera_health"]}])
    assert result[0]["passed"] is False
    assert "scope='camera_health'" in result[0]["failures"]
