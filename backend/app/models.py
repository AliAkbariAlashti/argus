import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, ARRAY

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
