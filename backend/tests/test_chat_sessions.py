from datetime import datetime, timezone
from unittest.mock import Mock

from app.main import _recent_history, _save_turn
from app.models import ChatSession


def test_save_turn_assigns_session_and_titles_first_question():
    session = ChatSession(id="session-1", title="New chat", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    db = Mock()
    db.get.return_value = session
    _save_turn(db, session.id, "user", "Check the loading entrance")
    message = db.add.call_args.args[0]
    assert message.session_id == "session-1"
    assert session.title == "Check the loading entrance"


def test_recent_history_is_scoped_to_session():
    db = Mock()
    query = db.query.return_value
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query
    query.all.return_value = []
    assert _recent_history(db, "session-2") == []
    query.filter.assert_called_once()
