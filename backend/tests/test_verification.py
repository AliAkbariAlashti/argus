import base64
import io
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from app.verification import select_candidates, verification_prompt, wants_verification


def _snapshot():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def test_verification_intent_is_explicit():
    assert wants_verification("Verify the person detected yesterday")
    assert not wants_verification("How many people were detected yesterday?")


def test_candidate_selection_requires_decodable_evidence(monkeypatch):
    camera = SimpleNamespace(id="cam1", name="Lobby", zone_tags=[])
    when = datetime(2026, 9, 10, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(id="one", event_id=1, camera_id="cam1", track_id="track1", confidence=.9,
                        observed_at=when, object_type="person", kind="object_detection",
                        to_dict=lambda: {"observed_at": when.isoformat()}),
        SimpleNamespace(id="repeat", event_id=2, camera_id="cam1", track_id="track1", confidence=.8,
                        observed_at=when, object_type="person", kind="object_detection",
                        to_dict=lambda: {"observed_at": when.isoformat()}),
    ]
    query = Mock()
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query
    query.all.return_value = rows
    db = Mock()
    db.get.side_effect = [SimpleNamespace(snapshot=_snapshot()), SimpleNamespace(snapshot=_snapshot())]
    monkeypatch.setattr("app.verification.observation_query", lambda *args: query)
    monkeypatch.setattr("app.verification.plan_observation_query", lambda *args: {"intent": "list"})

    plan, selected = select_candidates(db, "Verify people detected yesterday", [camera])
    assert plan == {"intent": "list"}
    assert [row.id for row, _ in selected] == ["one"]
    assert "camera=Lobby" in verification_prompt("Verify", selected, {"cam1": "Lobby"})


def test_chat_verification_sends_only_selected_evidence(monkeypatch):
    from app import main

    camera = SimpleNamespace(id="cam1", name="Lobby", zone_tags=[], created_at=None)
    when = datetime(2026, 9, 10, tzinfo=timezone.utc)
    observation = SimpleNamespace(
        id="obs1", event_id=10, camera_id="cam1", observed_at=when,
        object_type="person", confidence=.9, kind="object_detection",
        to_dict=lambda: {"observed_at": when.isoformat()},
    )
    db = Mock()
    db.query.return_value.order_by.return_value.all.return_value = [camera]
    ask = Mock(return_value="The Lobby image supports a person detection.")
    monkeypatch.setattr(main, "vlm", SimpleNamespace(ready=True, error=None, ask=ask))
    monkeypatch.setattr(
        main, "select_candidates",
        lambda *args: ({"intent": "list"}, [(observation, Image.new("RGB", (8, 8)))]),
    )
    monkeypatch.setattr(main, "_recent_history", lambda db: [])
    monkeypatch.setattr(main, "_save_turn", Mock())

    result = main._answer_chat(main.ChatRequest(question="Verify the person detected yesterday"), db)
    assert result["scope"] == "verification"
    assert result["evidence"][0]["observation_id"] == "obs1"
    assert len(ask.call_args.args[0]) == 1
