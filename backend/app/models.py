import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, ARRAY, Integer, Boolean, JSON, Float, Index, UniqueConstraint

from .db import Base


def _utc_iso(value):
    if value is None:
        return None
    # Existing timestamp columns store UTC without a timezone. Include the
    # offset in the API so browsers do not interpret them as local time.
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, default=_new_id)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False, default="")
    zone_tags = Column(ARRAY(String), nullable=False, default=list)
    description = Column(Text, nullable=False, default="")

    # "file" (looping local video) or "rtsp" (live network stream)
    source_type = Column(String, nullable=False, default="file")
    source_path = Column(String, nullable=False)  # filename under VIDEOS_DIR, or rtsp url

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "zone_tags": self.zone_tags or [],
            "description": self.description,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "created_at": _utc_iso(self.created_at),
        }


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    title = Column(String(120), nullable=False, default="New chat")
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {"id": self.id, "title": self.title, "created_at": _utc_iso(self.created_at), "updated_at": _utc_iso(self.updated_at)}


class ChatMessage(Base):
    """One turn of the (single, fleet-wide) conversation. Persisted so the
    transcript survives a refresh and can be replayed to the model as
    context on later questions."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(32), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" | "assistant"
    text = Column(Text, nullable=False)
    # Which cameras informed this answer (empty for user turns).
    cameras_used = Column(ARRAY(String), nullable=False, default=list)
    # Snapshot of the frame(s) the answer was based on, as a data URI, so the
    # UI can show what the model actually saw. Null for user turns.
    snapshot = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "text": self.text,
            "cameras_used": self.cameras_used or [],
            "snapshot": self.snapshot,
            "created_at": _utc_iso(self.created_at),
        }


class AgentToolRun(Base):
    """Audit record for each model-selected Argus tool invocation."""

    __tablename__ = "agent_tool_runs"
    __table_args__ = (Index("ix_agent_tool_runs_session_time", "session_id", "created_at"),)

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    session_id = Column(String(32), nullable=False)
    step = Column(Integer, nullable=False)
    tool = Column(String(80), nullable=False)
    arguments = Column(JSON, nullable=False, default=dict)
    result = Column(JSON, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="completed")
    duration_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {"id": self.id, "session_id": self.session_id, "step": self.step, "tool": self.tool,
                "arguments": self.arguments or {}, "result": self.result or {}, "status": self.status,
                "duration_ms": self.duration_ms, "created_at": _utc_iso(self.created_at)}


class Event(Base):
    """A system-detected occurrence — from the background VLM monitor or CPU
    tools' own detectors (motion, face, people, object). This is the single,
    full log across every camera and every kind of detection; the Activity
    page is a search/filter view over this table, with alert rules as one of
    the filter dimensions rather than a gate on what gets logged at all.

    `confidence` is set for object detections (from YOLO) and used to decide
    which snapshot survives when the same detection repeats back-to-back
    (see cpu.py's dedup logic) — otherwise null."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False)
    severity = Column(String, nullable=False, default="info")  # "info" | "warning" | "critical"
    category = Column(String, nullable=False)
    summary = Column(Text, nullable=False, default="")
    snapshot = Column(Text, nullable=True)
    confidence = Column(Integer, nullable=True)  # stored as an integer percent (0-100); null when not applicable
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "category": self.category,
            "summary": self.summary,
            "snapshot": self.snapshot,
            "confidence": self.confidence,
            "created_at": _utc_iso(self.created_at),
        }


class Observation(Base):
    """Structured, machine-queryable fact produced by a perception worker.

    IDs are generated by the producer instead of the database so observations
    from multiple sites or edge nodes can be merged without sequence clashes.
    Boxes are normalized [x, y, width, height] values and remain independent of
    camera resolution or decoder implementation.
    """

    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_camera_time", "camera_id", "observed_at"),
        Index("ix_observations_object_time", "object_type", "observed_at"),
        Index("ix_observations_action_time", "action", "observed_at"),
        Index("ix_observations_track_time", "track_id", "observed_at"),
    )

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    camera_id = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    producer_id = Column(String, nullable=False, default="local")
    source = Column(String, nullable=False)  # cpu, vlm, external, operator
    kind = Column(String, nullable=False)  # object_detection, line_crossing, dwell
    object_type = Column(String, nullable=True)
    track_id = Column(String, nullable=True)  # globally unique track instance
    global_entity_id = Column(String, nullable=True)  # reserved for cross-camera association
    confidence = Column(Float, nullable=True)  # normalized 0..1
    box = Column(JSON, nullable=True)  # normalized [x, y, width, height]
    zone = Column(String, nullable=True)
    action = Column(String, nullable=True)
    direction = Column(String, nullable=True)
    dwell_seconds = Column(Float, nullable=True)
    event_id = Column(Integer, nullable=True)  # optional link to operator-facing activity
    attributes = Column(JSON, nullable=False, default=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "observed_at": _utc_iso(self.observed_at),
            "recorded_at": _utc_iso(self.recorded_at),
            "producer_id": self.producer_id,
            "source": self.source,
            "kind": self.kind,
            "object_type": self.object_type,
            "track_id": self.track_id,
            "global_entity_id": self.global_entity_id,
            "confidence": self.confidence,
            "box": self.box,
            "zone": self.zone,
            "action": self.action,
            "direction": self.direction,
            "dwell_seconds": self.dwell_seconds,
            "event_id": self.event_id,
            "attributes": self.attributes or {},
        }


class VisualEmbedding(Base):
    """Portable visual vector attached to a structured observation.

    JSON storage works with the project's supported databases. Deployments can
    move similarity search to a native vector index later without changing the
    producer contract or regenerating observation IDs.
    """

    __tablename__ = "visual_embeddings"
    __table_args__ = (
        UniqueConstraint("observation_id", "provider", "model_version", name="uq_visual_embedding_version"),
        Index("ix_visual_embeddings_camera_time", "camera_id", "observed_at"),
        Index("ix_visual_embeddings_object_time", "object_type", "observed_at"),
    )

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    observation_id = Column(String(32), nullable=False)
    camera_id = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    producer_id = Column(String, nullable=False, default="local")
    object_type = Column(String, nullable=True)
    provider = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    dimensions = Column(Integer, nullable=False)
    vector = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self, include_vector=False):
        result = {
            "id": self.id,
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "observed_at": _utc_iso(self.observed_at),
            "producer_id": self.producer_id,
            "object_type": self.object_type,
            "provider": self.provider,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "content_hash": self.content_hash,
            "created_at": _utc_iso(self.created_at),
        }
        if include_vector:
            result["vector"] = self.vector
        return result


class CameraLink(Base):
    """Directed travel-time constraint between two camera views."""

    __tablename__ = "camera_links"
    __table_args__ = (UniqueConstraint("from_camera_id", "to_camera_id", name="uq_camera_link_direction"),)

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    from_camera_id = Column(String, nullable=False)
    to_camera_id = Column(String, nullable=False)
    min_travel_seconds = Column(Float, nullable=False, default=0)
    max_travel_seconds = Column(Float, nullable=False)
    description = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id, "from_camera_id": self.from_camera_id, "to_camera_id": self.to_camera_id,
            "min_travel_seconds": self.min_travel_seconds, "max_travel_seconds": self.max_travel_seconds,
            "description": self.description, "created_at": _utc_iso(self.created_at),
        }


class EntityAssociation(Base):
    """Auditable cross-camera match proposal or operator decision."""

    __tablename__ = "entity_associations"
    __table_args__ = (
        UniqueConstraint("source_observation_id", "target_observation_id", name="uq_entity_association_pair"),
        Index("ix_entity_associations_status_created", "status", "created_at"),
    )

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    source_observation_id = Column(String(32), nullable=False)
    target_observation_id = Column(String(32), nullable=False)
    source_track_id = Column(String, nullable=False)
    target_track_id = Column(String, nullable=False)
    similarity = Column(Float, nullable=False)
    travel_seconds = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="suggested")
    global_entity_id = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    decided_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self):
        return {
            "id": self.id, "source_observation_id": self.source_observation_id,
            "target_observation_id": self.target_observation_id, "source_track_id": self.source_track_id,
            "target_track_id": self.target_track_id, "similarity": self.similarity,
            "travel_seconds": self.travel_seconds, "status": self.status,
            "global_entity_id": self.global_entity_id, "created_at": _utc_iso(self.created_at),
            "decided_at": _utc_iso(self.decided_at),
        }


class AlertRule(Base):
    """Operator-defined watch condition. Two independent sources:

    - "vlm": free-text target ("a person carrying a backpack"), judged by
      the connected vision model. Every active vlm rule for a camera folds
      into one shared structured classification call (see
      grounding.build_event_prompt and monitor.py) instead of one extra
      VLM call per rule — custom rules don't multiply GPU load. Requires a
      connected, ready vision model; a no-op otherwise.
    - "cpu": target is one of YOLO_CLASSES (see cpu.py), matched exactly
      against the CPU-tools object detector. Works with no vision model at
      all — every install, GPU or not.
    """

    __tablename__ = "alert_rules"

    id = Column(String, primary_key=True, default=_new_id)
    camera_id = Column(String, nullable=True)  # null = every camera
    source = Column(String, nullable=False, default="vlm")  # "vlm" | "cpu"
    target = Column(String, nullable=False)
    # Severity the logged Event gets when this rule matches. "vlm" rules are
    # always "warning" (see grounding.py); "cpu" rules choose their own, since
    # a "car" match and a "knife" match shouldn't read the same in Activity.
    severity = Column(String, nullable=False, default="info")  # "info" | "warning" | "critical"
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "source": self.source,
            "target": self.target,
            "severity": self.severity,
            "enabled": self.enabled,
            "created_at": _utc_iso(self.created_at),
        }


class CpuConfig(Base):
    __tablename__ = "cpu_configs"
    camera_id = Column(String, primary_key=True)
    settings = Column(JSON, nullable=False, default=dict)
