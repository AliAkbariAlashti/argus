import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, ARRAY, Integer, Boolean, JSON

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

    # "file" (looping local video, MVP) or "rtsp" (coming soon)
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


class ChatMessage(Base):
    """One turn of the (single, fleet-wide) conversation. Persisted so the
    transcript survives a refresh and can be replayed to the model as
    context on later questions."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
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
            "role": self.role,
            "text": self.text,
            "cameras_used": self.cameras_used or [],
            "snapshot": self.snapshot,
            "created_at": _utc_iso(self.created_at),
        }


class Event(Base):
    """A system-detected occurrence from the background monitor — not a
    user chat turn. Logged only on a state change (something entering or
    leaving view, a weapon becoming visible), so the table stays a
    meaningful timeline instead of one row per monitor poll. Feeds the
    Alerts tab."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String, nullable=False)
    severity = Column(String, nullable=False, default="info")  # "info" | "warning" | "critical"
    category = Column(String, nullable=False)
    summary = Column(Text, nullable=False, default="")
    snapshot = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "category": self.category,
            "summary": self.summary,
            "snapshot": self.snapshot,
            "created_at": _utc_iso(self.created_at),
        }


class AlertRule(Base):
    """Operator-defined watch condition ("car", "backpack left behind"). The
    background monitor folds every active rule for a camera into its regular
    structured classification call (see grounding.build_event_prompt and
    monitor.py) instead of one extra VLM call per rule — custom rules don't
    multiply GPU load."""

    __tablename__ = "alert_rules"

    id = Column(String, primary_key=True, default=_new_id)
    camera_id = Column(String, nullable=True)  # null = every camera
    target = Column(String, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "target": self.target,
            "enabled": self.enabled,
            "created_at": _utc_iso(self.created_at),
        }


class CpuConfig(Base):
    __tablename__ = "cpu_configs"
    camera_id = Column(String, primary_key=True)
    settings = Column(JSON, nullable=False, default=dict)
