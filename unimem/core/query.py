"""Query objects + ``QueryBuilder`` + ``QueryResult``.

A ``Query`` is the *only* read-side argument the framework passes to modules.
It carries three orthogonal filter axes (semantic / spatial / temporal), an
optional ``slot_filter`` (used by the graph to restrict fan-out), a free-form
``text`` for modules that want to do their own embedding/NLP, and a ``top_k``
hint.

``QueryResult`` is what each module returns from ``read``: the entries it
matched plus an optional score list and provenance fields filled in by the
graph layer.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from ..typing import SemanticKey, SpatialKey
from .entry import MemoryEntry
from .slots import MemorySlot


class Query:
    """A multi-axis, slot-aware read request."""

    __slots__ = (
        "text",
        "semantic",
        "spatial",
        "t_min",
        "t_max",
        "slot_filter",
        "top_k",
        "metadata",
    )

    def __init__(
        self,
        text: Optional[str] = None,
        semantic: Optional[Sequence[SemanticKey]] = None,
        spatial: Optional[Sequence[Sequence[float]]] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        slot_filter: Optional[Iterable[Any]] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.text = text
        self.semantic: List[SemanticKey] = list(semantic) if semantic else []
        self.spatial: List[SpatialKey] = (
            [tuple(float(c) for c in k) for k in spatial] if spatial else []
        )
        self.t_min = float(t_min) if t_min is not None else None
        self.t_max = float(t_max) if t_max is not None else None
        # Normalise slot_filter to a set of MemorySlot members.
        self.slot_filter = (
            {MemorySlot.from_value(s) for s in slot_filter}
            if slot_filter is not None
            else None
        )
        self.top_k = int(top_k) if top_k is not None else None
        self.metadata: dict = dict(metadata) if metadata else {}

    # -- dunder ------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Query(text={self.text!r}, sem={self.semantic}, spa={self.spatial}, "
            f"t=[{self.t_min},{self.t_max}], slots={self.slot_filter}, k={self.top_k})"
        )

    # -- convenience ------------------------------------------------------- #
    @property
    def has_temporal(self) -> bool:
        return self.t_min is not None or self.t_max is not None

    @property
    def has_spatial(self) -> bool:
        return len(self.spatial) > 0

    @property
    def has_semantic(self) -> bool:
        return len(self.semantic) > 0

    def accepts_slot(self, slot: Any) -> bool:
        """True if this query targets the given slot (or all slots)."""
        if self.slot_filter is None:
            return True
        return MemorySlot.from_value(slot) in self.slot_filter


class QueryBuilder:
    """Fluent builder for :class:`Query`."""

    def __init__(self) -> None:
        self._q = Query()

    def with_text(self, text: str) -> "QueryBuilder":
        self._q.text = text
        return self

    def with_semantic(self, *keys: SemanticKey) -> "QueryBuilder":
        self._q.semantic.extend(keys)
        return self

    def with_spatial(self, *keys: Sequence[float]) -> "QueryBuilder":
        for k in keys:
            self._q.spatial.append(tuple(float(c) for c in k))
        return self

    def with_temporal(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> "QueryBuilder":
        if t_min is not None:
            self._q.t_min = float(t_min)
        if t_max is not None:
            self._q.t_max = float(t_max)
        return self

    def with_slot(self, *slots: Any) -> "QueryBuilder":
        if self._q.slot_filter is None:
            self._q.slot_filter = set()
        for s in slots:
            self._q.slot_filter.add(MemorySlot.from_value(s))
        return self

    def with_top_k(self, k: int) -> "QueryBuilder":
        self._q.top_k = int(k)
        return self

    def with_metadata(self, **kv: Any) -> "QueryBuilder":
        self._q.metadata.update(kv)
        return self

    def build(self) -> Query:
        return Query(
            text=self._q.text,
            semantic=list(self._q.semantic),
            spatial=list(self._q.spatial),
            t_min=self._q.t_min,
            t_max=self._q.t_max,
            slot_filter=set(self._q.slot_filter) if self._q.slot_filter is not None else None,
            top_k=self._q.top_k,
            metadata=dict(self._q.metadata),
        )


class QueryResult:
    """What one module returns for one query.

    ``entries`` is ordered best-first when scores are provided. The graph
    layer fills in ``source_node_id`` / ``source_slot`` so callers can know
    which module produced which result.
    """

    __slots__ = ("entries", "scores", "source_node_id", "source_slot", "metadata")

    def __init__(
        self,
        entries: Optional[List[MemoryEntry]] = None,
        scores: Optional[List[float]] = None,
        source_node_id: Optional[str] = None,
        source_slot: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.entries: List[MemoryEntry] = list(entries) if entries else []
        self.scores: Optional[List[float]] = (
            list(scores) if scores is not None else None
        )
        self.source_node_id = source_node_id
        self.source_slot = source_slot
        self.metadata: dict = dict(metadata) if metadata else {}

    # -- dunder ------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QueryResult(n={len(self.entries)}, node={self.source_node_id!r}, "
            f"slot={self.source_slot!r})"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    # -- helpers ----------------------------------------------------------- #
    @classmethod
    def empty(cls) -> "QueryResult":
        return cls()

    def extend(self, other: "QueryResult") -> "QueryResult":
        """Append another result's entries; drops score alignment."""
        self.entries.extend(other.entries)
        if self.scores is not None and other.scores is not None:
            self.scores.extend(other.scores)
        else:
            self.scores = None
        return self

    def truncate(self, top_k: int) -> "QueryResult":
        """In-place truncation to the first ``top_k`` entries."""
        if top_k is not None and top_k >= 0 and len(self.entries) > top_k:
            self.entries = self.entries[:top_k]
            if self.scores is not None:
                self.scores = self.scores[:top_k]
        return self


__all__ = ["Query", "QueryBuilder", "QueryResult"]
