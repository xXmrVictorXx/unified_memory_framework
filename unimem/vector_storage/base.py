"""``VectorStorage`` — vector-database abstraction for unimem.

Used **on demand** by modules that need vector retrieval (R4's semantic
embedding search, video clip embeddings, ...). The generic
:class:`~unimem.core.module.MemoryModule` does **not** depend on a
VectorStorage — it's an explicit dependency for methods that need it.

Public API:

* :class:`VectorStorage` — the ABC every backend implements.
* :func:`create_vector_storage` — factory driven by a config dict.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple


class VectorStorage(ABC):
    """Abstract vector storage with collection-level isolation.

    Concrete implementations must override every ``@abstractmethod``. The
    semantics mirror Qdrant's API surface (collection / point / payload) but
    are intentionally minimal — most methods have straightforward equivalents
    in FAISS, Milvus, Chroma, pgvector, etc.
    """

    @abstractmethod
    def create_collection(
        self,
        name: str,
        vector_dim: int,
        distance_metric: str = "cosine",
    ) -> bool:
        """Idempotent collection creation. Returns True on success."""

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Drop a collection (and every point it holds)."""

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """True if the named collection has been created."""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: Sequence[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert (or replace) a single point."""

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Approximate nearest-neighbour search.

        Returns ``[(point_id, score, payload), ...]`` sorted best-first.
        ``score_threshold`` semantics depend on the metric (cosine: keep
        scores >= threshold; L2: keep scores <= threshold).
        ``filter_conditions`` is a flat ``{key: value}`` dict; only points
        whose payload matches all keys are considered.
        """

    @abstractmethod
    def get(
        self, collection: str, point_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Return ``(vector, payload)`` for ``point_id``, or None."""

    @abstractmethod
    def delete(self, collection: str, point_id: str) -> bool:
        """Delete a point. Returns True if the point existed."""

    @abstractmethod
    def scroll(
        self,
        collection: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """List ``(point_id, payload)`` tuples in insertion order."""

    @abstractmethod
    def count(self, collection: str) -> int:
        """Number of points in the collection."""


__all__ = ["VectorStorage"]
