from evals import run_agent_eval


def test_eval_requires_agent_scope_expected_tool_and_confirmation(monkeypatch):
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        if path == "/api/agent/status": return {"configured": True}
        if path == "/api/chat/sessions": return {"id": "eval-session"}
        return {"deleted": "eval-session"}

    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    monkeypatch.setattr(run_agent_eval, "stream_answer", lambda *args, **kwargs: {"scope": "agent", "answer": "Prepared.", "tool_trace": [{"tool": "propose_alert_rule"}], "pending_actions": [{"id": "action-1"}]})
    result = run_agent_eval.run("http://argus", [{"name": "proposal", "question": "create rule", "expected_tools": ["propose_alert_rule"], "expects_confirmation": True, "max_seconds": 5}])
    assert result[0]["passed"] is True


def test_eval_reports_deterministic_fallback_as_failure(monkeypatch):
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        if path == "/api/agent/status": return {"configured": True}
        if path == "/api/chat/sessions": return {"id": "eval-session"}
        return {}

    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    monkeypatch.setattr(run_agent_eval, "stream_answer", lambda *args, **kwargs: {"scope": "camera_health", "answer": "fallback", "tool_trace": []})
    result = run_agent_eval.run("http://argus", [{"name": "health", "question": "online?", "expected_tools": ["get_camera_health"]}])
    assert result[0]["passed"] is False
    assert "scope='camera_health'" in result[0]["failures"]


def test_eval_reuses_one_session_for_follow_up_turns(monkeypatch):
    created = []
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        if path == "/api/agent/status": return {"configured": True}
        if path == "/api/chat/sessions": created.append(1); return {"id": "same-session"}
        return {}
    payloads = []
    def fake_stream(base, payload, timeout=300, total_timeout=None):
        payloads.append(payload)
        return {"scope": "agent", "answer": "ok", "tool_trace": [{"tool": "list_cameras"}]}
    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    monkeypatch.setattr(run_agent_eval, "stream_answer", fake_stream)
    result = run_agent_eval.run("http://argus", [{"name": "follow-up", "turns": [
        {"question": "list", "expected_tools": ["list_cameras"]},
        {"question": "those?", "expected_tools": ["list_cameras"]},
    ]}])
    assert all(item["passed"] for item in result)
    assert created == [1]
    assert [item["session_id"] for item in payloads] == ["same-session", "same-session"]


def test_eval_accepts_any_equivalent_expected_tool():
    result = run_agent_eval.assess("health", {"scope": "agent", "answer": "ok", "tool_trace": [{"tool": "list_cameras"}]}, {"expected_any_tools": ["list_cameras", "get_camera_health"]}, 1)
    assert result["passed"] is True


def test_eval_rejects_confirmation_with_missing_required_arguments():
    answer = {"scope": "agent", "answer": "prepared", "tool_trace": [{"tool": "propose_alert_rule"}],
              "pending_actions": [{"action": "create_alert_rule", "arguments": {"target": ""}}]}
    result = run_agent_eval.assess("action", answer, {"expects_confirmation": True,
        "expected_action": "create_alert_rule", "required_action_arguments": ["target"]}, 1)
    assert result["passed"] is False
    assert "missing action argument target" in result["failures"]


def test_eval_checks_action_argument_meaning():
    answer = {"scope": "agent", "answer": "prepared", "tool_trace": [{"tool": "propose_alert_rule"}],
              "pending_actions": [{"action": "create_alert_rule", "arguments": {"target": "all cameras"}}]}
    result = run_agent_eval.assess("action", answer, {"action_argument_contains": {"target": "backpack"}}, 1)
    assert result["passed"] is False
    assert "action argument target does not contain backpack" in result["failures"]


def test_eval_checks_answer_content():
    answer = {"scope": "agent", "answer": "One or more camera IDs do not exist.", "tool_trace": []}
    result = run_agent_eval.assess("recordings", answer, {"answer_contains": ["recordings", "FPS"]}, 1)
    assert result["passed"] is False
    assert "answer does not contain recordings" in result["failures"]
    assert "answer does not contain FPS" in result["failures"]


def test_eval_checks_answer_exclusions():
    answer = {"scope": "agent", "answer": "Server Room and East Corridor are online.", "tool_trace": []}
    result = run_agent_eval.assess("scope", answer, {"answer_excludes": ["East Corridor"]}, 1)
    assert result["passed"] is False
    assert "answer unexpectedly contains East Corridor" in result["failures"]


def test_eval_approval_is_idempotent_and_cleans_up_created_rule(monkeypatch):
    calls = []
    approved = {"id": "action-1", "status": "approved", "result": {"id": "rule-1"}}
    def fake_request(base, path, method="GET", payload=None, timeout=180):
        calls.append((path, method, payload))
        return approved
    monkeypatch.setattr(run_agent_eval, "request", fake_request)
    answer = {"pending_actions": [{"id": "action-1", "action": "create_alert_rule"}]}
    assessment = {"passed": True, "failures": []}
    result = run_agent_eval.verify_approval("http://argus", answer,
        {"approve_action": True, "expected_action": "create_alert_rule"}, assessment)
    assert result["passed"] is True
    assert calls == [
        ("/api/agent/actions/action-1/decision", "POST", {"decision": "approve"}),
        ("/api/agent/actions/action-1/decision", "POST", {"decision": "approve"}),
        ("/api/alert-rules/rule-1", "DELETE", None),
    ]
