"""Replaceable visual descriptors and bounded similarity search.

The built-in provider is a low-cost appearance descriptor, useful for finding
objects with similar colour distribution. It is deliberately provider-labelled
so a deployment can add CLIP/SigLIP without pretending this vector is semantic.
"""
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .models import Observation, VisualEmbedding


@dataclass(frozen=True)
class Descriptor:
    provider: str
    model_version: str
    vector: list[float]
    content_hash: str


class CpuAppearanceProvider:
    name = "opencv-hsv"
    version = "spatial-8x4x4-v1"

    def describe(self, crop: np.ndarray) -> Descriptor:
        if crop is None or crop.size == 0:
            raise ValueError("Cannot describe an empty image crop.")
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        parts = []
        for y0, y1 in ((0, 32), (32, 64)):
            for x0, x1 in ((0, 32), (32, 64)):
                hist = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
                parts.append(hist.reshape(-1))
        vector = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return Descriptor(
            self.name,
            self.version,
            [round(float(value), 8) for value in vector],
            hashlib.sha256(resized.tobytes()).hexdigest(),
        )


default_provider = CpuAppearanceProvider()


def crop_normalized(frame: np.ndarray, box) -> np.ndarray:
    if frame is None or not box or len(box) != 4:
        raise ValueError("A frame and normalized box are required.")
    height, width = frame.shape[:2]
    x, y, box_width, box_height = (float(value) for value in box)
    left = max(0, min(width - 1, int(round(x * width))))
    top = max(0, min(height - 1, int(round(y * height))))
    right = max(left + 1, min(width, int(round((x + box_width) * width))))
    bottom = max(top + 1, min(height, int(round((y + box_height) * height))))
    return frame[top:bottom, left:right]


def record_visual_embedding(db, observation: Observation, frame: np.ndarray, provider=default_provider):
    descriptor = provider.describe(crop_normalized(frame, observation.box))
    row = VisualEmbedding(
        observation_id=observation.id,
        camera_id=observation.camera_id,
        observed_at=observation.observed_at,
        producer_id=observation.producer_id,
        object_type=observation.object_type,
        provider=descriptor.provider,
        model_version=descriptor.model_version,
        dimensions=len(descriptor.vector),
        vector=descriptor.vector,
        content_hash=descriptor.content_hash,
    )
    db.add(row)
    return row


def _cosine(left, right):
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def rank_embeddings(reference_vector, rows, limit=20):
    ranked = []
    for row in rows:
        score = _cosine(reference_vector, row.vector)
        if score is not None:
            ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return ranked[:max(1, min(limit, 100))]


def find_similar(db, observation_id, *, camera_id=None, object_type=None, since: datetime | None = None, until: datetime | None = None, limit=20):
    reference = db.query(VisualEmbedding).filter(VisualEmbedding.observation_id == observation_id).order_by(VisualEmbedding.created_at.desc()).first()
    if reference is None:
        return None
    query = db.query(VisualEmbedding).filter(
        VisualEmbedding.provider == reference.provider,
        VisualEmbedding.model_version == reference.model_version,
        VisualEmbedding.observation_id != observation_id,
    )
    if camera_id:
        query = query.filter(VisualEmbedding.camera_id == camera_id)
    if object_type:
        query = query.filter(VisualEmbedding.object_type == object_type.strip().lower())
    if since:
        query = query.filter(VisualEmbedding.observed_at >= since)
    if until:
        query = query.filter(VisualEmbedding.observed_at < until)
    maximum = max(1, int(os.environ.get("ARGUS_VISUAL_SEARCH_MAX_CANDIDATES", "5000")))
    candidates = query.order_by(VisualEmbedding.observed_at.desc()).limit(maximum + 1).all()
    truncated = len(candidates) > maximum
    ranked = rank_embeddings(reference.vector, candidates[:maximum], limit)
    return {
        "reference": reference.to_dict(),
        "matches": [{**row.to_dict(), "similarity": round(score, 6)} for score, row in ranked],
        "candidate_count": min(len(candidates), maximum),
        "candidates_truncated": truncated,
    }
