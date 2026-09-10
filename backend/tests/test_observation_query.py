from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Event, Observation
from app.observation_query import answer_observation_question, plan_observation_query


CAMERAS = [
    SimpleNamespace(id="cam1", name="Server Room", zone_tags=["server-room"], created_at=None),
    SimpleNamespace(id="cam2", name="Front Door", zone_tags=["entrance"], created_at=None),
]


def test_planner_leaves_current_visual_questions_for_live_vision():
    assert plan_observation_query("How many people are there right now?", CAMERAS) is None


def test_planner_extracts_history_scope_object_group_and_action():
    plan = plan_observation_query("Which cameras saw vehicles crossing today?", CAMERAS)
    assert plan["intent"] == "cameras"
    assert set(plan["object_types"]) == {"bicycle", "car", "motorcycle", "bus", "truck"}
    assert plan["kind"] == "line_crossing"
    assert plan["since"] is not None


def test_yesterday_has_a_closed_time_window():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    plan = plan_observation_query("people detected yesterday", CAMERAS, now=now)
    assert plan["since"] == "2026-09-09T00:00:00+00:00"
    assert plan["until"] == "2026-09-10T00:00:00+00:00"


def test_answer_counts_unique_tracks_instead_of_repeated_samples():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Event.__table__.create(engine)
    Observation.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    now = datetime.now(timezone.utc)
    with Session() as db:
        event = Event(camera_id="cam1", category="CPU · Person detected", summary="person", snapshot="data:image/jpeg;base64,test")
        db.add(event)
        db.flush()
        db.add_all([
            Observation(camera_id="cam1", observed_at=now, source="cpu", kind="object_detection", object_type="person", track_id="track-a", action="observed", event_id=event.id),
            Observation(camera_id="cam1", observed_at=now, source="cpu", kind="object_detection", object_type="person", track_id="track-a", action="observed", event_id=event.id),
            Observation(camera_id="cam2", observed_at=now, source="cpu", kind="object_detection", object_type="person", track_id="track-b", action="observed", event_id=event.id),
        ])
        db.commit()
        result = answer_observation_question(db, "How many people were detected today?", CAMERAS)
        assert result["answer"] == "I found 2 unique tracked people."
        assert set(result["cameras_used"]) == {"cam1", "cam2"}
        assert result["snapshot"] == "data:image/jpeg;base64,test"
