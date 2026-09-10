from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.models import Observation
from app.visual_search import (
    CpuAppearanceProvider, Descriptor, crop_normalized, find_by_text, initialize_vector_backend,
    rank_embeddings, record_visual_embedding, vector_backend_status,
)


def test_cpu_descriptor_is_deterministic_and_normalized():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = (20, 100, 220)
    provider = CpuAppearanceProvider()
    first = provider.describe(image)
    second = provider.describe(image.copy())
    assert first == second
    assert len(first.vector) == 512
    assert np.isclose(np.linalg.norm(first.vector), 1.0)


def test_crop_normalized_uses_resolution_independent_box():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_normalized(frame, [0.25, 0.2, 0.5, 0.4]).shape == (40, 100, 3)


def test_embedding_is_linked_to_observation_and_provider_version():
    db = Mock()
    observation = Observation(
        id="obs-1", camera_id="cam-1", producer_id="edge-1", object_type="car",
        observed_at=None, box=[0.0, 0.0, 1.0, 1.0],
    )
    row = record_visual_embedding(db, observation, np.zeros((20, 20, 3), dtype=np.uint8))
    assert row.observation_id == "obs-1"
    assert row.provider == "opencv-hsv"
    assert row.model_version == "spatial-8x4x4-v1"
    assert row.dimensions == 512
    db.add.assert_called_once_with(row)


def test_similarity_ranking_is_deterministic_and_skips_wrong_dimensions():
    rows = [
        SimpleNamespace(id="same", vector=[1.0, 0.0]),
        SimpleNamespace(id="different", vector=[0.0, 1.0]),
        SimpleNamespace(id="wrong", vector=[1.0]),
    ]
    ranked = rank_embeddings([1.0, 0.0], rows)
    assert [row.id for _, row in ranked] == ["same", "different"]
    assert ranked[0][0] == 1.0


def test_pgvector_request_falls_back_on_non_postgres(monkeypatch):
    monkeypatch.setenv("ARGUS_VECTOR_BACKEND", "pgvector")
    engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
    status = initialize_vector_backend(engine)
    assert status["requested"] == "pgvector"
    assert status["active"] == "json"
    assert status["ready"] is False
    assert "requires PostgreSQL" in status["error"]


def test_json_backend_is_default(monkeypatch):
    monkeypatch.delenv("ARGUS_VECTOR_BACKEND", raising=False)
    status = initialize_vector_backend(SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    assert status == {"requested": "json", "active": "json", "ready": True, "error": None}
    assert vector_backend_status() == status


def test_text_search_uses_shared_semantic_space(monkeypatch):
    from app import visual_search

    provider = SimpleNamespace(describe_text=lambda query: Descriptor("clip", "model-v1", [1.0, 0.0], "hash"))
    rows = [
        SimpleNamespace(id="best", vector=[1.0, 0.0], to_dict=lambda: {"id": "best"}),
        SimpleNamespace(id="weak", vector=[0.0, 1.0], to_dict=lambda: {"id": "weak"}),
    ]
    query = Mock()
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query
    query.all.return_value = rows
    db = Mock()
    db.query.return_value = query
    monkeypatch.setattr(visual_search, "get_provider", lambda: provider)

    result = find_by_text(db, "red vehicle", limit=2)
    assert result["provider"] == "clip"
    assert [match["id"] for match in result["matches"]] == ["best", "weak"]
    assert result["matches"][0]["similarity"] == 1.0
