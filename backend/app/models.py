import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, ARRAY, Integer

from .db import Base


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
            "created_at": self.created_at.isoformat() if self.created_at else None,
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
