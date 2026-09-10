from types import SimpleNamespace
from unittest.mock import Mock

from app import main


def test_historical_chat_uses_observation_plan_without_calling_vlm(monkeypatch):
    camera = SimpleNamespace(id="cam1", name="Server Room", zone_tags=[], created_at=None)
    db = Mock()
    db.query.return_value.order_by.return_value.all.return_value = [camera]
    saved = []
    monkeypatch.setattr(main, "_save_turn", lambda db, role, text, **kwargs: saved.append((role, text, kwargs)))
    monkeypatch.setattr(main, "answer_observation_question", lambda *args: {
        "answer": "I found 2 unique tracked people.",
        "cameras_used": ["cam1"],
        "snapshot": None,
        "plan": {"intent": "count"},
    })
    monkeypatch.setattr(main.vlm, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM must not run")))

    result = main._answer_chat(main.ChatRequest(question="How many people were detected today?"), db)

    assert result["scope"] == "observations"
    assert result["answer"] == "I found 2 unique tracked people."
    assert [turn[0] for turn in saved] == ["user", "assistant"]
