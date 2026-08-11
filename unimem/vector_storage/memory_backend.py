"""``InMemoryVectorStorage`` — brute-force cosine / L2 search, pure stdlib.

Default fallback used by tests and any unimem module that wants vector
retrieval without an external dependency. Scaling is O(N) per query — fine
for unit tests and small dev datasets; production deployments should wire
up Qdrant (or another real ANN backend).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import VectorStorage


class _Collection:
    """A named bag of (id → vector, payload)."""

    def __init__(self, vector_dim: int, distance_metric: str) -> None:
        self.vector_dim = int(vector_dim)
        self.distance_metric = str(distance_metric).lower()
        if self.distance_metric not in ("cosine", "l2", "dot"):
            raise ValueError(
                f"distance_metric must be 'cosine', 'l2', or 'dot', got "
                f"{self.distance_metric!r}"
            )
        # Insertion-ordered id → (vector, payload)
        self._points: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    def upsert(
        self,
        point_id: str,
        vector: Sequence[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        vec_list = [float(x) for x in vector]
        if len(vec_list) != self.vector_dim:
            raise ValueError(
                f"vector dim mismatch: expected {self.vector_dim}, got {len(vec_list)}"
            )
        self._points[point_id] = (vec_list, dict(payload or {}))

    def get(
        self, point_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        item = self._points.get(point_id)
        if item is None:
            return None
        # Return copies so callers can't mutate state.
        return (list(item[0]), dict(item[1]))

    def delete(self, point_id: str) -> bool:
        return self._points.pop(point_id, None) is not None

    def count(self) -> int:
        return len(self._points)

    def items(self) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        return [
            (pid, list(vec), dict(payload))
            for pid, (vec, payload) in self._points.items()
        ]

    # -- distance helpers ------------------------------------------------- #
    def score(self, query: Sequence[float], vec: Sequence[float]) -> float:
        if self.distance_metric == "cosine":
            return _cosine(query, vec)
        if self.distance_metric == "dot":
            return _dot(query, vec)
        # l2 — return *negative* distance so higher is better (consistent sort)
        return -_l2(query, vec)

    def keep(self, score: float, threshold: Optional[float]) -> bool:
        if threshold is None:
            return True
        if self.distance_metric == "l2":
            # score is negative distance; keep if distance <= threshold
            return -score <= threshold
        # cosine / dot: keep if score >= threshold
        return score >= threshold


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _matches_filter(payload: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
    for k, v in conditions.items():
        if payload.get(k) != v:
            return False
    return True


class InMemoryVectorStorage(VectorStorage):
    """In-memory vector DB (brute force). Pure stdlib."""

    def __init__(self) -> None:
        self._collections: Dict[str, _Collection] = {}

    # ------------------------------------------------------------------ #
    # Collection management
    # ------------------------------------------------------------------ #
    def create_collection(
        self,
        name: str,
        vector_dim: int,
        distance_metric: str = "cosine",
    ) -> bool:
        if name in self._collections:
            # Idempotent: leave the existing collection alone.
            return True
        self._collections[name] = _Collection(vector_dim, distance_metric)
        return True

    def delete_collection(self, name: str) -> bool:
        return self._collections.pop(name, None) is not None

    def collection_exists(self, name: str) -> bool:
        return name in self._collections

    # ------------------------------------------------------------------ #
    # Point CRUD
    # ------------------------------------------------------------------ #
    def upsert(
        self,
        collection: str,
        point_id: str,
        vector: Sequence[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        col = self._require_collection(collection)
        col.upsert(point_id, vector, payload)
        return True

    def get(
        self, collection: str, point_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        col = self._collections.get(collection)
        if col is None:
            return None
        return col.get(point_id)

    def delete(self, collection: str, point_id: str) -> bool:
        col = self._collections.get(collection)
        if col is None:
            return False
        return col.delete(point_id)

    def scroll(
        self,
        collection: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        col = self._collections.get(collection)
        if col is None:
            return []
        out: List[Tuple[str, Dict[str, Any]]] = []
        for pid, _, payload in col.items():
            if filter_conditions and not _matches_filter(payload, filter_conditions):
                continue
            out.append((pid, payload))
            if len(out) >= limit:
                break
        return out

    def count(self, collection: str) -> int:
        col = self._collections.get(collection)
        return col.count() if col is not None else 0

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        col = self._collections.get(collection)
        if col is None:
            return []
        q = [float(x) for x in query_vector]
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for pid, vec, payload in col.items():
            if filter_conditions and not _matches_filter(payload, filter_conditions):
                continue
            s = col.score(q, vec)
            if not col.keep(s, score_threshold):
                continue
            scored.append((pid, s, dict(payload)))
        scored.sort(key=lambda x: -x[1])  # higher score = better
        return scored[:top_k]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_collection(self, name: str) -> _Collection:
        if name not in self._collections:
            raise KeyError(
                f"Unknown vector collection: {name!r}. "
                f"Call create_collection() first."
            )
        return self._collections[name]


__all__ = ["InMemoryVectorStorage"]
