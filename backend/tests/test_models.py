from datetime import datetime

from app.models import Camera


def test_utc_database_timestamp_is_explicit_in_api():
    camera = Camera(id="test", name="Test", source_path="test.mp4", created_at=datetime(2026, 9, 9, 8, 0))
    assert camera.to_dict()["created_at"] == "2026-09-09T08:00:00+00:00"
