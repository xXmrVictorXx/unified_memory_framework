"""Write-side gating policies.

A ``WritePolicy`` decides whether a (module, entry, context) tuple should be
admitted. Edge-level policies model INHerit-SG-style *event-triggered writes*
(only ingest on topological change); module-level policies model per-memory
concerns (e.g. a WM that always overwrites vs. an SM that only adds novel
facts).

The ``DedupPolicy`` (R4's Eq. 5) generalises the same hook into a
semantic+spatial de-duplication gate backed by a :class:`VectorStorage`
for ANN search.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry
from ..core.module import MemoryModule

if TYPE_CHECKING:
    from ..graph_storage.base import GraphStorage
    from ..vector_storage.base import VectorStorage


class WritePolicy(ABC):
    """Decide whether ``module`` should store ``entry`` at ``context``."""

    @abstractmethod
    def should_write(
        self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext
    ) -> bool:
        """Return True to admit, False to skip (and stop graph propagation)."""


class AlwaysWrite(WritePolicy):
    """Admit everything. Default for FEEDS edges and root nodes."""

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return True


class NeverWrite(WritePolicy):
    """Admit nothing. Useful for stub / read-only modules."""

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return False


class LambdaWritePolicy(WritePolicy):
    """Wrap a plain callable as a :class:`WritePolicy` (handy in tests/scripts)."""

    def __init__(self, fn: Any) -> None:
        if not callable(fn):
            raise TypeError("LambdaWritePolicy expects a callable")
        self._fn = fn

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return bool(self._fn(module, entry, context))


# --------------------------------------------------------------------------- #
# DedupPolicy — R4 Eq. 5
# --------------------------------------------------------------------------- #
class DedupPolicy(WritePolicy):
    """R4 Eq. 5 deduplication as a :class:`WritePolicy`.

    Given a new ``entry``:

    1. Compute its embedding via ``embedding_fn``.
    2. Search the bound :class:`VectorStorage` collection for top-k
       semantically similar candidates (cosine ≥ ``delta_s``).
    3. For each candidate, fetch its spatial centroid from the
       :class:`GraphStorage` node properties; if the Euclidean distance to
       the new entry's centroid is < ``eps_c``, **merge** into the existing
       node (update properties in place) and return False — i.e. "don't
       write a new node".
    4. Otherwise (no match) upsert the embedding into the vector store and
       return True (allow the module to write a new node).

    The new entry's centroid is read from ``entry.spatial_keys[0][:3]``;
    the candidate's centroid is read from the GraphStorage node's
    ``spatial_keys[0][:3]`` property.

    Parameters
    ----------
    vector_storage:
        Pre-configured :class:`VectorStorage` (e.g. Qdrant).
    graph_storage:
        The backing :class:`GraphStorage` (used for spatial property lookup
        and merge).
    collection:
        Vector collection name.
    embedding_fn:
        Callable ``(text: str) -> Sequence[float]``.
    eps_c:
        Spatial distance threshold (Euclidean metres).
    delta_s:
        Semantic similarity threshold (cosine, in [-1, 1]).
    vector_dim:
        Embedding dimension (used to create the collection lazily).
    """

    def __init__(
        self,
        vector_storage: "VectorStorage",
        graph_storage: "GraphStorage",
        collection: str,
        embedding_fn: Callable[[str], Sequence[float]],
        eps_c: float = 0.5,
        delta_s: float = 0.7,
        vector_dim: Optional[int] = None,
        distance_metric: str = "cosine",
    ) -> None:
        self._vs = vector_storage
        self._gs = graph_storage
        self._collection = collection
        self._embedding_fn = embedding_fn
        self.eps_c = float(eps_c)
        self.delta_s = float(delta_s)
        self._vector_dim = vector_dim
        self._distance_metric = distance_metric
        # Lazily create the vector collection on first use.
        self._collection_ready = False

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_ready:
            return
        if not self._vs.collection_exists(self._collection):
            actual_dim = self._vector_dim or dim
            self._vs.create_collection(
                self._collection,
                vector_dim=actual_dim,
                distance_metric=self._distance_metric,
            )
        self._collection_ready = True

    def should_write(
        self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext
    ) -> bool:
        # 1. Embedding
        text = entry.text or ""
        if not text:
            # Without text we can't dedup semantically — admit.
            return True
        emb = list(self._embedding_fn(text))
        self._ensure_collection(len(emb))

        # 2. ANN search for semantically similar candidates
        results = self._vs.search(
            self._collection,
            emb,
            top_k=10,
            score_threshold=self.delta_s,
        )

        # 3. Spatial dedup
        new_centroid = self._extract_centroid(entry)
        if new_centroid is not None:
            for cand_id, score, payload in results:
                cand_centroid = self._candidate_centroid(cand_id, payload)
                if cand_centroid is None:
                    continue
                dist = _euclidean(new_centroid, cand_centroid)
                if dist < self.eps_c:
                    # Merge: update the existing node + vector payload.
                    self._merge(cand_id, entry, emb, new_centroid)
                    return False

        # 4. Admit and remember the embedding.
        payload = {
            "entry_id": entry.entry_id,
            "text": text,
        }
        if new_centroid is not None:
            payload["centroid"] = list(new_centroid)
        self._vs.upsert(self._collection, entry.entry_id, emb, payload)
        return True

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_centroid(entry: MemoryEntry):
        if not entry.spatial_keys:
            return None
        spa = entry.spatial_keys[0]
        if len(spa) >= 3:
            return tuple(float(c) for c in spa[:3])
        return None

    def _candidate_centroid(self, cand_id: str, payload: dict):
        # Prefer payload-stored centroid; fall back to GraphStorage lookup.
        c = payload.get("centroid")
        if c and len(c) >= 3:
            return tuple(float(x) for x in c[:3])
        node = self._gs.get_node(cand_id)
        if node is None:
            return None
        sk = node["properties"].get("spatial_keys") or []
        if sk and len(sk[0]) >= 3:
            return tuple(float(x) for x in sk[0][:3])
        return None

    def _merge(
        self,
        candidate_id: str,
        new_entry: MemoryEntry,
        new_embedding: Sequence[float],
        new_centroid,
    ) -> None:
        """Update the candidate node + vector payload in place.

        Strategy: keep the candidate's id, refresh its centroid to the
        latest observation, append the new timestamp if provided, and
        re-upsert the vector with updated payload.
        """
        node = self._gs.get_node(candidate_id)
        if node is None:
            return
        props = dict(node["properties"])
        # Spatial: latest wins
        if new_centroid is not None:
            sk = props.get("spatial_keys") or []
            if sk:
                sk[0] = list(new_centroid) + list(sk[0][3:])
            else:
                sk = [list(new_centroid)]
            props["spatial_keys"] = sk
        # Temporal: append unique timestamps
        if new_entry.temporal_keys:
            tk = list(props.get("temporal_keys") or [])
            for t in new_entry.temporal_keys:
                if float(t) not in tk:
                    tk.append(float(t))
            props["temporal_keys"] = tk
        # Metadata: merge
        meta = dict(props.get("metadata") or {})
        meta.update(new_entry.metadata)
        props["metadata"] = meta
        # Update labels (preserve existing)
        self._gs.add_node(candidate_id, list(node["labels"]), props)
        # Update vector payload
        payload = {
            "entry_id": candidate_id,
            "text": props.get("text") or new_entry.text,
            "centroid": list(new_centroid) if new_centroid is not None else None,
        }
        self._vs.upsert(self._collection, candidate_id, list(new_embedding), payload)


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


__all__ = [
    "WritePolicy",
    "AlwaysWrite",
    "NeverWrite",
    "LambdaWritePolicy",
    "DedupPolicy",
]
