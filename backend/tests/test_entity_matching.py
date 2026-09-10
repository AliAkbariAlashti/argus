from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from app.entity_matching import decide_association, travel_seconds


def test_travel_constraint_accepts_only_configured_window():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    link = SimpleNamespace(min_travel_seconds=10, max_travel_seconds=60)
    assert travel_seconds(start, start + timedelta(seconds=30), link) == 30
    assert travel_seconds(start, start + timedelta(seconds=5), link) is None
    assert travel_seconds(start, start + timedelta(seconds=61), link) is None


def test_confirmed_association_assigns_one_global_entity():
    association = SimpleNamespace(
        status="suggested", decided_at=None, global_entity_id=None,
        source_track_id="track-a", target_track_id="track-b",
    )
    source = SimpleNamespace(global_entity_id=None)
    target = SimpleNamespace(global_entity_id=None)
    first_query, second_query = Mock(), Mock()
    first_query.filter.return_value.all.return_value = [source]
    second_query.filter.return_value.all.return_value = [target]
    db = Mock()
    db.query.side_effect = [first_query, second_query]

    decide_association(db, association, "confirmed")
    assert association.status == "confirmed"
    assert source.global_entity_id == target.global_entity_id == association.global_entity_id
    assert len(association.global_entity_id) == 32
    db.commit.assert_called_once()


def test_conflicting_entities_are_not_merged():
    association = SimpleNamespace(source_track_id="track-a", target_track_id="track-b")
    first_query, second_query = Mock(), Mock()
    first_query.filter.return_value.all.return_value = [SimpleNamespace(global_entity_id="entity-a")]
    second_query.filter.return_value.all.return_value = [SimpleNamespace(global_entity_id="entity-b")]
    db = Mock()
    db.query.side_effect = [first_query, second_query]
    try:
        decide_association(db, association, "confirmed")
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("Conflicting entities must not be merged")
    db.commit.assert_not_called()
