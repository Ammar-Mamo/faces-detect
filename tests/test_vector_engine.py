from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from engines.vector_engine import VectorEngine


def make_vectors(count: int, dimension: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, dimension)).astype(np.float32)


def test_search_returns_matching_face_ids_with_similarity_scores() -> None:
    engine = VectorEngine(dimension=4)

    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = [101, 102, 103]
    engine.add(vectors, ids)

    matches = engine.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), threshold=0.8, top_k=3)

    assert len(matches) == 2
    assert matches[0].face_id == 101
    assert abs(matches[0].score - 1.0) < 1e-5
    assert matches[1].face_id == 102
    assert abs(matches[1].score - 0.9938837) < 1e-5
    assert matches[0].score > matches[1].score


def test_save_and_load_index_round_trip(tmp_path: Path) -> None:
    engine = VectorEngine(dimension=8)

    vectors = make_vectors(count=128, dimension=8)
    ids = np.arange(1000, 1128, dtype=np.int64)
    engine.add(vectors, ids)

    index_path = tmp_path / "index.faiss"
    saved_path = engine.save(index_path)

    loaded_engine = VectorEngine.load(saved_path)
    query = vectors[17]

    original_matches = engine.search(query, threshold=0.2, top_k=5)
    loaded_matches = loaded_engine.search(query, threshold=0.2, top_k=5)

    assert saved_path == index_path
    assert saved_path.exists()
    assert loaded_engine.ntotal == engine.ntotal
    assert loaded_matches == original_matches


def test_search_handles_thousands_of_embeddings_quickly() -> None:
    engine = VectorEngine(dimension=128)

    vectors = make_vectors(count=5000, dimension=128)
    ids = np.arange(1, 5001, dtype=np.int64)
    engine.add(vectors, ids)

    query = vectors[4321]

    started_at = time.perf_counter()
    matches = engine.search(query, threshold=0.3, top_k=20)
    elapsed = time.perf_counter() - started_at

    assert matches
    assert matches[0].face_id == 4322
    assert matches[0].score > 0.99
    assert elapsed < 2.0
