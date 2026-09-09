from datetime import datetime, timedelta, timezone

from app.history import parse_question

CAMERAS = [
    {"id": "cam1", "name": "Server Room", "zone_tags": ["server-room", "data-center", "restricted"]},
    {"id": "cam2", "name": "East Corridor", "zone_tags": ["corridor", "hallway", "indoor"]},
]


def test_last_seen_question_resolves_camera_and_limit():
    parsed = parse_question("when did you last see a person on server room", CAMERAS)
    assert parsed["camera_id"] == "cam1"
    assert parsed["q"] == "person"
    assert parsed["limit"] == 1


def test_plain_listing_question_has_higher_limit():
    parsed = parse_question("show me car events on east corridor", CAMERAS)
    assert parsed["camera_id"] == "cam2"
    assert parsed["q"] == "car"
    assert parsed["limit"] == 50


def test_zone_tag_matches_camera():
    parsed = parse_question("any activity in the data center today", CAMERAS)
    assert parsed["camera_id"] == "cam1"


def test_relative_time_window_last_hour():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    parsed = parse_question("last hour on server room", CAMERAS, now=now)
    since = datetime.fromisoformat(parsed["since"])
    assert since == now - timedelta(hours=1)


def test_numeric_relative_time_window():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    parsed = parse_question("events in the last 3 days", CAMERAS, now=now)
    since = datetime.fromisoformat(parsed["since"])
    assert since == now - timedelta(days=3)


def test_severity_extracted():
    parsed = parse_question("any critical events today", CAMERAS)
    assert parsed["severity"] == "critical"


def test_severity_word_not_also_used_as_keyword():
    # Regression: "critical" matched as severity AND left in as a keyword
    # meant results needed "critical" literally in category/summary text,
    # which never happens — the query always returned nothing.
    parsed = parse_question("any critical events today", CAMERAS)
    assert parsed["q"] is None


def test_no_camera_match_leaves_camera_id_none():
    parsed = parse_question("when did you last see a dog", CAMERAS)
    assert parsed["camera_id"] is None
    assert parsed["q"] == "dog"


def test_no_keyword_left_is_none():
    parsed = parse_question("what happened on server room today", CAMERAS)
    assert parsed["q"] is None
