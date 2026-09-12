from types import SimpleNamespace
from unittest.mock import Mock

from app.agent_actions import create_proposal, decide_action
from app.models import AgentAction
from app.models import AlertRule


def test_create_proposal_does_not_apply_change():
    db = Mock()
    proposal = create_proposal(db, "session-1", "create_alert_rule", {"target": "person"}, "Create person alert")
    assert isinstance(proposal, AgentAction)
    assert proposal.status == "pending"
    assert proposal.arguments == {"target": "person"}
    db.add.assert_called_once_with(proposal)


def test_reject_proposal_is_persisted_without_running_services():
    row = AgentAction(id="action-1", session_id="session-1", action="delete_camera", summary="Delete Lobby", arguments={"camera_id": "cam-1"}, status="pending")
    db = Mock()
    db.refresh.side_effect = lambda value: None
    result = decide_action(db, row, "reject", SimpleNamespace(), SimpleNamespace())
    assert result.status == "rejected"
    assert result.decided_at is not None
    db.commit.assert_called_once()


def test_decision_retry_is_idempotent():
    row = AgentAction(id="action-1", session_id="session-1", action="delete_camera", summary="Delete Lobby", arguments={}, status="approved", result={"deleted": "cam-1"})
    db = Mock()
    result = decide_action(db, row, "approve", SimpleNamespace(), SimpleNamespace())
    assert result is row
    db.commit.assert_not_called()


def test_approved_rule_update_applies_only_declared_fields():
    proposal = AgentAction(id="action-2", session_id="session-1", action="update_alert_rule", summary="Disable rule", arguments={"rule_id": "rule-1", "enabled": False}, status="pending")
    rule = AlertRule(id="rule-1", source="cpu", target="person", severity="info", enabled=True)
    db = Mock()
    db.get.side_effect = lambda model, key: rule if model is AlertRule else proposal
    db.refresh.side_effect = lambda value: None
    result = decide_action(db, proposal, "approve", SimpleNamespace(), SimpleNamespace())
    assert result.status == "approved"
    assert rule.enabled is False
    assert rule.severity == "info"
