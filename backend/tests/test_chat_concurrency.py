from app.main import _conversation_lock


def test_conversations_have_independent_locks():
    first = _conversation_lock("session-a")
    same = _conversation_lock("session-a")
    other = _conversation_lock("session-b")
    assert first is same
    assert first is not other
    assert first.acquire(blocking=False)
    try:
        assert not same.acquire(blocking=False)
        assert other.acquire(blocking=False)
        other.release()
    finally:
        first.release()
