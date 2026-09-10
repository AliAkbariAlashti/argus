from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from app.observations import record_observation


def test_observation_writer_uses_portable_ids_boxes_and_node_identity(monkeypatch):
    monkeypatch.setenv("ARGUS_NODE_ID", "edge-north")
    db = Mock()
    observed_at = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    row = record_observation(
        db,
        camera_id="cam-1",
        kind="object_detection",
        observed_at=observed_at,
        object_type="person",
        track_id="track-1",
        confidence=0.9,
        box=[0.1, 0.2, 0.3, 0.4],
        action="observed",
    )
    assert db.add.call_args.args[0] is row
    assert row.producer_id == "edge-north"
    assert row.observed_at == observed_at
    assert row.box == [0.1, 0.2, 0.3, 0.4]


def test_observation_writer_rejects_non_normalized_boxes():
    with pytest.raises(ValueError, match="normalized"):
        record_observation(Mock(), camera_id="cam-1", kind="object_detection", box=[10, 20, 30, 40])
