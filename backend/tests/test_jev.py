from app import jev as jev_module
from app.jev import JevIntegration


TOOLS = [{"function": {"name": "list_cameras", "description": "List camera records."}}]
ALERT_TOOL = {
    "function": {
        "name": "propose_alert_rule",
        "description": "Prepare an alert rule.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"},
            "source": {"type": "string", "enum": ["cpu", "vlm"]},
            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        }},
    },
}


def _choice(choice, confidence=0.9):
    return {
        "model": "jev-test",
        "answers": {"route": {
            "choice": choice,
            "confidence": confidence,
            "probabilities": {choice: confidence, "needs_planner": 1 - confidence},
        }},
    }


def _active(monkeypatch, threshold=0.65):
    monkeypatch.setattr(jev_module, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev_module, "JEV_ROUTING_MODE", "active")
    monkeypatch.setattr(jev_module, "JEV_ROUTE_THRESHOLD", threshold)
    instance = JevIntegration()
    monkeypatch.setattr(instance, "_record_route", lambda job, **kwargs: "run-1")
    return instance


def test_active_route_selects_only_a_registered_tool(monkeypatch):
    integration = _active(monkeypatch)
    requests = []
    monkeypatch.setattr(integration, "_request", lambda body: requests.append(body) or _choice("list_cameras"))

    route = integration.route_request(question="List the cameras", session_id="session-1", tools=TOOLS)

    assert route["decision"] == "routed"
    assert route["suggested_tool"] == "list_cameras"
    assert route["run_id"] == "run-1"
    assert requests[0]["questions"]["route"]["criteria"]["list_cameras"] == "List camera records."
    assert requests[0]["questions"]["route"]["criteria"]["needs_planner"]
    assert requests[0]["questions"]["route"]["criteria"]["direct_answer"]


def test_active_route_falls_back_below_confidence_threshold(monkeypatch):
    integration = _active(monkeypatch, threshold=0.7)
    monkeypatch.setattr(integration, "_request", lambda _body: _choice("list_cameras", confidence=0.6))

    route = integration.route_request(question="Maybe list cameras", session_id="session-1", tools=TOOLS)

    assert route["decision"] == "planner_fallback"
    assert route["suggested_tool"] == "list_cameras"


def test_active_route_sends_multi_step_requests_to_planner(monkeypatch):
    integration = _active(monkeypatch)
    monkeypatch.setattr(integration, "_request", lambda _body: _choice("needs_planner"))

    route = integration.route_request(question="Find a person, then inspect the camera", session_id="session-1", tools=TOOLS)

    assert route["decision"] == "planner_fallback"


def test_active_route_fails_open_to_planner(monkeypatch):
    integration = _active(monkeypatch)
    monkeypatch.setattr(integration, "_request", lambda _body: (_ for _ in ()).throw(TimeoutError()))

    route = integration.route_request(question="List cameras", session_id="session-1", tools=TOOLS)

    assert route["decision"] == "service_fallback"
    assert route["reason"]


def test_router_is_disabled_without_key(monkeypatch):
    monkeypatch.setattr(jev_module, "TYPESAFE_API_KEY", "")
    monkeypatch.setattr(jev_module, "JEV_ROUTING_MODE", "active")
    integration = JevIntegration()

    assert integration.route_request(question="List cameras", session_id="session-1", tools=TOOLS) is None


def test_active_route_fills_only_explicit_closed_arguments(monkeypatch):
    integration = _active(monkeypatch)
    responses = iter([
        _choice("propose_alert_rule"),
        {"model": "jev-test", "answers": {
            "arg_source": {"choice": "vlm", "confidence": 0.94},
            "arg_severity": {"choice": "__unspecified__", "confidence": 0.91},
        }},
    ])
    bodies = []
    monkeypatch.setattr(integration, "_request", lambda body: bodies.append(body) or next(responses))

    route = integration.route_request(
        question="Create a VLM warning for a person carrying a bag",
        session_id="session-1",
        tools=[ALERT_TOOL],
    )

    assert route["decision"] == "routed"
    assert route["arguments"] == {"source": "vlm"}
    assert route["argument_confidences"] == {"source": 0.94, "severity": 0.91}
    assert set(bodies[1]["questions"]) == {"arg_source", "arg_severity"}
    assert "__unspecified__" in bodies[1]["questions"]["arg_severity"]["criteria"]


def test_uncertain_closed_argument_falls_back_to_qwen(monkeypatch):
    integration = _active(monkeypatch, threshold=0.7)
    responses = iter([
        _choice("propose_alert_rule"),
        {"answers": {
            "arg_source": {"choice": "vlm", "confidence": 0.6},
            "arg_severity": {"choice": "__unspecified__", "confidence": 0.9},
        }},
    ])
    monkeypatch.setattr(integration, "_request", lambda _body: next(responses))

    route = integration.route_request(
        question="Create an alert for a person",
        session_id="session-1",
        tools=[ALERT_TOOL],
    )

    assert route["decision"] == "planner_fallback"
    assert route["arguments"] == {}
    assert "uncertain" in route["reason"]
