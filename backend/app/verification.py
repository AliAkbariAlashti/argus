"""Select a small evidence set for historical VLM verification."""
import base64
import io
import os
import re

from PIL import Image

from .models import Event, Observation
from .observation_query import observation_query, plan_observation_query

_VERIFY_RE = re.compile(r"\b(verify|confirm|double[- ]?check|check the evidence|prove)\b", re.IGNORECASE)


def wants_verification(question):
    return bool(_VERIFY_RE.search(question))


def _snapshot_image(value):
    if not value or not value.startswith("data:image/") or "," not in value:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1], validate=True))).convert("RGB")
    except Exception:
        return None


def select_candidates(db, question, cameras, forced_camera_id=None):
    plan = plan_observation_query(question, cameras, forced_camera_id)
    if plan is None:
        return None, []
    maximum = max(1, min(int(os.environ.get("ARGUS_VLM_VERIFY_MAX_FRAMES", "6")), 12))
    # Read a bounded shortlist, then prefer high confidence while preserving
    # camera and track diversity. This avoids spending all VLM inputs on one
    # object's repeated detector samples.
    rows = (
        observation_query(db, plan)
        .filter(Observation.event_id.isnot(None))
        .order_by(Observation.observed_at.desc(), Observation.id.desc())
        .limit(maximum * 20)
        .all()
    )
    candidates = []
    for row in rows:
        event = db.get(Event, row.event_id)
        image = _snapshot_image(event.snapshot if event else None)
        if image is not None:
            candidates.append((row, image))
    candidates.sort(key=lambda item: (-(item[0].confidence or 0), -item[0].observed_at.timestamp(), item[0].id))
    selected, tracks, cameras_seen = [], set(), set()
    for row, image in candidates:
        identity = row.track_id or row.id
        if identity in tracks:
            continue
        # First pass favours a new camera. Remaining slots are filled below.
        if row.camera_id in cameras_seen and len(cameras_seen) < len({item[0].camera_id for item in candidates}):
            continue
        selected.append((row, image))
        tracks.add(identity)
        cameras_seen.add(row.camera_id)
        if len(selected) == maximum:
            break
    if len(selected) < maximum:
        selected_ids = {row.id for row, _ in selected}
        for row, image in candidates:
            identity = row.track_id or row.id
            if row.id in selected_ids or identity in tracks:
                continue
            selected.append((row, image))
            tracks.add(identity)
            if len(selected) == maximum:
                break
    return plan, selected


def verification_prompt(question, selected, camera_names):
    evidence = []
    for index, (row, _) in enumerate(selected, 1):
        evidence.append(
            f"Image {index}: camera={camera_names.get(row.camera_id, row.camera_id)}; "
            f"timestamp={row.to_dict()['observed_at']}; detector_object={row.object_type or 'unknown'}; "
            f"confidence={row.confidence if row.confidence is not None else 'unknown'}; event={row.kind}."
        )
    return (
        "Verify the user's historical question using only the candidate evidence images below. "
        "Detector labels are retrieval hints and can be wrong. State what the images support, "
        "what they do not prove, and cite camera names and timestamps. Do not claim identity "
        "across images.\n\n" + "\n".join(evidence) + f"\n\nQuestion: {question}"
    )
