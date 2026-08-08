from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


class VectorEngineError(RuntimeError):
    """Base class for vector indexing errors."""


class VectorDimensionError(VectorEngineError):
    """Raised when vectors do not match the configured embedding dimension."""


@dataclass(frozen=True)
class SearchMatch:
    face_id: int
    score: float


class VectorEngine:
    def __init__(self, dimension: int, *, index: faiss.Index | None = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = dimension
        self._index = index or self._create_index(dimension)

        if self._index.d != dimension:
            raise VectorDimensionError(
                f"Index dimension mismatch: expected {dimension}, got {self._index.d}"
            )

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int]) -> None:
        normalized_vectors = self._prepare_vectors(vectors)
        prepared_ids = self._prepare_ids(ids, normalized_vectors.shape[0])
        self._index.add_with_ids(normalized_vectors, prepared_ids)

    def search(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.0,
        *,
        top_k: int = 10,
    ) -> list[SearchMatch]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.ntotal == 0:
            return []

        query = self._prepare_query_vector(query_vector)
        scores, ids = self._index.search(query, min(top_k, self.ntotal))

        matches: list[SearchMatch] = []
        for score, face_id in zip(scores[0], ids[0]):
            if face_id == -1:
                continue
            if float(score) < threshold:
                continue
            matches.append(SearchMatch(face_id=int(face_id), score=float(score)))
        return matches

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(output_path))
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> VectorEngine:
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        index = faiss.read_index(str(index_path))
        return cls(index.d, index=index)

    @staticmethod
    def _create_index(dimension: int) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise VectorDimensionError("vectors must be a 2D array or a single 1D vector")
        if array.shape[1] != self.dimension:
            raise VectorDimensionError(
                f"Vector dimension mismatch: expected {self.dimension}, got {array.shape[1]}"
            )
        return self._normalize(array)

    def _prepare_query_vector(self, vector: np.ndarray) -> np.ndarray:
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise VectorDimensionError("query_vector must be a single 1D vector or shape (1, dimension)")
        if query.shape[1] != self.dimension:
            raise VectorDimensionError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape[1]}"
            )
        return self._normalize(query)

    @staticmethod
    def _prepare_ids(ids: np.ndarray | list[int], expected_size: int) -> np.ndarray:
        prepared = np.asarray(ids, dtype=np.int64).reshape(-1)
        if prepared.shape[0] != expected_size:
            raise ValueError(
                f"ids length mismatch: expected {expected_size}, got {prepared.shape[0]}"
            )
        return prepared

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        normalized = np.ascontiguousarray(vectors.copy())
        norms = np.linalg.norm(normalized, axis=1)
        if np.any(norms == 0):
            raise ValueError("zero-length vectors are not supported")
        faiss.normalize_L2(normalized)
        return normalized
