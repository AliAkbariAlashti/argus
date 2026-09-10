"""Conservative cross-camera track association."""
import os
import uuid
from datetime import timedelta, timezone

from .models import CameraLink, EntityAssociation, Observation, VisualEmbedding
from .visual_search import _cosine


def _utc(value):
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def travel_seconds(source_time, target_time, link):
    elapsed = (_utc(target_time) - _utc(source_time)).total_seconds()
    return elapsed if link.min_travel_seconds <= elapsed <= link.max_travel_seconds else None


def suggest_matches(db, observation_id, limit=10):
    source = db.get(Observation, observation_id)
    if source is None or not source.track_id:
        return None
    source_embedding = (
        db.query(VisualEmbedding)
        .filter(VisualEmbedding.observation_id == source.id)
        .order_by(VisualEmbedding.created_at.desc())
        .first()
    )
    if source_embedding is None:
        return {"source": source.to_dict(), "matches": [], "reason": "Source observation has no visual embedding."}
    links = db.query(CameraLink).filter(CameraLink.from_camera_id == source.camera_id).all()
    threshold = min(1.0, max(0.0, float(os.environ.get("ARGUS_ENTITY_MATCH_MIN_SIMILARITY", "0.80"))))
    proposals = {}
    for link in links:
        start = _utc(source.observed_at) + timedelta(seconds=link.min_travel_seconds)
        end = _utc(source.observed_at) + timedelta(seconds=link.max_travel_seconds)
        rows = (
            db.query(VisualEmbedding, Observation)
            .join(Observation, Observation.id == VisualEmbedding.observation_id)
            .filter(
                VisualEmbedding.camera_id == link.to_camera_id,
                VisualEmbedding.provider == source_embedding.provider,
                VisualEmbedding.model_version == source_embedding.model_version,
                VisualEmbedding.dimensions == source_embedding.dimensions,
                Observation.object_type == source.object_type,
                Observation.track_id.isnot(None),
                Observation.observed_at >= start,
                Observation.observed_at <= end,
            )
            .order_by(Observation.observed_at)
            .limit(max(100, limit * 20))
            .all()
        )
        for embedding, observation in rows:
            elapsed = travel_seconds(source.observed_at, observation.observed_at, link)
            if elapsed is None or observation.track_id == source.track_id:
                continue
            similarity = _cosine(source_embedding.vector, embedding.vector)
            if similarity is None or similarity < threshold:
                continue
            existing = proposals.get(observation.track_id)
            candidate = (similarity, observation, elapsed)
            if existing is None or candidate[0] > existing[0]:
                proposals[observation.track_id] = candidate
    ranked = sorted(proposals.values(), key=lambda item: (-item[0], item[2], item[1].id))[:max(1, min(limit, 50))]
    matches = []
    for similarity, target, travel_seconds in ranked:
        association = (
            db.query(EntityAssociation)
            .filter(EntityAssociation.source_observation_id == source.id, EntityAssociation.target_observation_id == target.id)
            .first()
        )
        if association is None:
            association = EntityAssociation(
                id=uuid.uuid4().hex, source_observation_id=source.id, target_observation_id=target.id,
                source_track_id=source.track_id, target_track_id=target.track_id,
                similarity=similarity, travel_seconds=travel_seconds,
            )
            db.add(association)
        matches.append({**association.to_dict(), "target": target.to_dict()})
    if matches:
        db.commit()
    return {"source": source.to_dict(), "matches": matches, "minimum_similarity": threshold}


def decide_association(db, association, status):
    if status not in ("confirmed", "rejected"):
        raise ValueError("Decision must be confirmed or rejected.")
    from datetime import datetime
    association.status = status
    association.decided_at = datetime.now(timezone.utc)
    if status == "confirmed":
        source_rows = db.query(Observation).filter(Observation.track_id == association.source_track_id).all()
        target_rows = db.query(Observation).filter(Observation.track_id == association.target_track_id).all()
        known = {row.global_entity_id for row in source_rows + target_rows if row.global_entity_id}
        if len(known) > 1:
            raise ValueError("Tracks already belong to conflicting confirmed entities.")
        entity_id = next(iter(known), uuid.uuid4().hex)
        for row in source_rows + target_rows:
            row.global_entity_id = entity_id
        association.global_entity_id = entity_id
    db.commit()
    return association
