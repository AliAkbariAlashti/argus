"""Portable structured observation writer shared by perception pipelines."""
import os
import uuid
from datetime import datetime, timezone

from .models import Observation


def _normalized_box(box):
    if box is None:
        return None
    if len(box) != 4:
        raise ValueError("An observation box must contain x, y, width, and height.")
    values = [round(float(value), 6) for value in box]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Observation boxes must use normalized values between 0 and 1.")
    if values[0] + values[2] > 1.000001 or values[1] + values[3] > 1.000001:
        raise ValueError("Observation box must fit inside the frame.")
    return values


def record_observation(
    db,
    *,
    camera_id,
    kind,
    source="cpu",
    observed_at=None,
    object_type=None,
    track_id=None,
    global_entity_id=None,
    confidence=None,
    box=None,
    zone=None,
    action=None,
    direction=None,
    dwell_seconds=None,
    event_id=None,
    attributes=None,
):
    """Add one observation to the caller's transaction.

    The caller owns commit/rollback so the structured fact and its optional
    operator-facing Event are atomic. ARGUS_NODE_ID distinguishes producers
    when several edge nodes write to one database.
    """
    observed_at = observed_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    row = Observation(
        id=uuid.uuid4().hex,
        camera_id=camera_id,
        observed_at=observed_at,
        producer_id=os.environ.get("ARGUS_NODE_ID", "local"),
        source=source,
        kind=kind,
        object_type=object_type,
        track_id=str(track_id) if track_id is not None else None,
        global_entity_id=global_entity_id,
        confidence=float(confidence) if confidence is not None else None,
        box=_normalized_box(box),
        zone=zone,
        action=action,
        direction=direction,
        dwell_seconds=float(dwell_seconds) if dwell_seconds is not None else None,
        event_id=event_id,
        attributes=attributes or {},
    )
    db.add(row)
    return row
