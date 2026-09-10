"""Replaceable visual descriptors and bounded similarity search.

The built-in provider is a low-cost appearance descriptor, useful for finding
objects with similar colour distribution. It is deliberately provider-labelled
so a deployment can add CLIP/SigLIP without pretending this vector is semantic.
"""
import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from sqlalchemy import text

from .models import Observation, VisualEmbedding

log = logging.getLogger("argus.visual_search")
_vector_backend = {"requested": "json", "active": "json", "ready": True, "error": None}
_provider_lock = threading.Lock()
_provider = None
_provider_state = {"requested": "opencv", "active": None, "ready": None, "error": None}


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


class TransformersClipProvider:
    """Optional shared image/text embedding space backed by Transformers."""

    name = "transformers-clip"

    def __init__(self, model_id=None):
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id or os.environ.get("ARGUS_SEMANTIC_MODEL", "openai/clip-vit-base-patch32")
        self.version = self.model_id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()

    @staticmethod
    def _descriptor(name, version, values, content):
        vector = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return Descriptor(name, version, [round(float(value), 8) for value in vector], hashlib.sha256(content).hexdigest())

    def describe(self, crop: np.ndarray) -> Descriptor:
        from PIL import Image
        import torch

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            values = self.model.get_image_features(**inputs)
        return self._descriptor(self.name, self.version, values, crop.tobytes())

    def describe_text(self, query: str) -> Descriptor:
        import torch

        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        with torch.inference_mode():
            values = self.model.get_text_features(**inputs)
        return self._descriptor(self.name, self.version, values, query.encode())


def get_provider():
    global _provider
    requested = os.environ.get("ARGUS_VISUAL_EMBEDDING_PROVIDER", "opencv").strip().lower()
    with _provider_lock:
        if _provider is not None and _provider_state["requested"] == requested:
            return _provider
        _provider_state.update(requested=requested, active=None, ready=False, error=None)
        try:
            if requested == "opencv":
                _provider = CpuAppearanceProvider()
            elif requested in ("clip", "transformers-clip"):
                _provider = TransformersClipProvider()
            else:
                raise ValueError(f"Unknown visual embedding provider: {requested}")
            _provider_state.update(active=_provider.name, ready=True)
            return _provider
        except Exception as exc:
            _provider = None
            _provider_state["error"] = str(exc)
            raise


def provider_status(load=False):
    if load and _provider_state["ready"] is None:
        try:
            get_provider()
        except Exception:
            pass
    return dict(_provider_state)


def initialize_vector_backend(engine):
    """Enable pgvector when requested, otherwise retain portable JSON search."""
    requested = os.environ.get("ARGUS_VECTOR_BACKEND", "json").strip().lower()
    _vector_backend.update(requested=requested, active="json", ready=True, error=None)
    if requested in ("", "json"):
        return vector_backend_status()
    if requested != "pgvector":
        _vector_backend.update(ready=False, error=f"Unknown vector backend: {requested}")
        return vector_backend_status()
    if engine.dialect.name != "postgresql":
        _vector_backend.update(ready=False, error="pgvector requires PostgreSQL; JSON fallback is active.")
        return vector_backend_status()
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("ALTER TABLE visual_embeddings ADD COLUMN IF NOT EXISTS native_vector vector"))
        _vector_backend.update(active="pgvector")
    except Exception as exc:
        _vector_backend.update(ready=False, error=f"pgvector unavailable; JSON fallback is active: {exc}")
        log.warning("%s", _vector_backend["error"])
    return vector_backend_status()


def vector_backend_status():
    return dict(_vector_backend)


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


def record_visual_embedding(db, observation: Observation, frame: np.ndarray, provider=None):
    provider = provider or get_provider()
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
    if _vector_backend["active"] == "pgvector":
        db.flush()
        db.execute(
            text("UPDATE visual_embeddings SET native_vector = CAST(:vector AS vector) WHERE id = :id"),
            {"id": row.id, "vector": "[" + ",".join(str(value) for value in descriptor.vector) + "]"},
        )
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


def find_by_text(db, query_text, *, camera_id=None, object_type=None, since=None, until=None, limit=20):
    provider = get_provider()
    if not hasattr(provider, "describe_text"):
        raise ValueError("Text search requires the optional CLIP embedding provider.")
    descriptor = provider.describe_text(query_text)
    query = db.query(VisualEmbedding).filter(
        VisualEmbedding.provider == descriptor.provider,
        VisualEmbedding.model_version == descriptor.model_version,
        VisualEmbedding.dimensions == len(descriptor.vector),
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
    ranked = rank_embeddings(descriptor.vector, candidates[:maximum], limit)
    return {
        "query": query_text,
        "provider": descriptor.provider,
        "model_version": descriptor.model_version,
        "matches": [{**row.to_dict(), "similarity": round(score, 6)} for score, row in ranked],
        "candidate_count": min(len(candidates), maximum),
        "candidates_truncated": truncated,
        "search_backend": "json",
    }


def find_similar(db, observation_id, *, camera_id=None, object_type=None, since: datetime | None = None, until: datetime | None = None, limit=20):
    reference = db.query(VisualEmbedding).filter(VisualEmbedding.observation_id == observation_id).order_by(VisualEmbedding.created_at.desc()).first()
    if reference is None:
        return None
    if _vector_backend["active"] == "pgvector":
        native = _find_similar_pgvector(
            db, reference, camera_id=camera_id, object_type=object_type,
            since=since, until=until, limit=limit,
        )
        if native is not None:
            return native
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
        "search_backend": "json",
    }


def _find_similar_pgvector(db, reference, *, camera_id=None, object_type=None, since=None, until=None, limit=20):
    clauses = [
        "candidate.id != reference.id",
        "candidate.native_vector IS NOT NULL",
        "reference.native_vector IS NOT NULL",
        "candidate.provider = reference.provider",
        "candidate.model_version = reference.model_version",
        "candidate.dimensions = reference.dimensions",
    ]
    params = {"observation_id": reference.observation_id, "limit": max(1, min(limit, 100))}
    for value, clause, key in (
        (camera_id, "candidate.camera_id = :camera_id", "camera_id"),
        (object_type.strip().lower() if object_type else None, "candidate.object_type = :object_type", "object_type"),
        (since, "candidate.observed_at >= :since", "since"),
        (until, "candidate.observed_at < :until", "until"),
    ):
        if value is not None:
            clauses.append(clause)
            params[key] = value
    statement = text(f"""
        SELECT candidate.id,
               1 - (candidate.native_vector <=> reference.native_vector) AS similarity
        FROM visual_embeddings AS candidate
        JOIN visual_embeddings AS reference
          ON reference.observation_id = :observation_id
        WHERE {' AND '.join(clauses)}
        ORDER BY candidate.native_vector <=> reference.native_vector, candidate.id
        LIMIT :limit
    """)
    scored = db.execute(statement, params).all()
    if not scored:
        return None
    scores = {row.id: float(row.similarity) for row in scored}
    rows = db.query(VisualEmbedding).filter(VisualEmbedding.id.in_(scores)).all()
    rows.sort(key=lambda row: (-scores[row.id], row.id))
    return {
        "reference": reference.to_dict(),
        "matches": [{**row.to_dict(), "similarity": round(scores[row.id], 6)} for row in rows],
        "candidate_count": len(rows),
        "candidates_truncated": False,
        "search_backend": "pgvector",
    }
