"""Multi-axis indexable memory entries.

``MemoryEntry`` is the unit of data that flows between memory modules. It is
intentionally language-anchored: ``text`` is the primary human/LLM-readable
handle, while ``semantic_keys`` / ``spatial_keys`` / ``temporal_keys`` are
secondary indices inspired by R4's three-axis object record. ``payload`` is an
opaque slot for anything the framework should not inspect (embeddings,
features, image hashes, bounding boxes, ...).

``MultiAxisIndex`` is an *optional utility* (not part of any ABC contract) that
maintains three inverted indices keyed by the three axis types. Reference
implementations (e.g. ``ListEpisodicMemory``) use it; modules with their own
native structure (e.g. a scene-graph tree) may ignore it entirely.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..typing import Metadata, Payload, SemanticKey, SpatialKey, TemporalKey


# --------------------------------------------------------------------------- #
# MemoryEntry
# --------------------------------------------------------------------------- #
class MemoryEntry:
    """A single indexable memory record.

    Attributes are intentionally mutable so policies and consolidation passes
    can annotate / extend entries in place. ``entry_id`` is the only field
    expected to remain stable after creation.
    """

    __slots__ = (
        "entry_id",
        "text",
        "payload",
        "semantic_keys",
        "spatial_keys",
        "temporal_keys",
        "metadata",
        "source_slot",
    )

    def __init__(
        self,
        entry_id: str,
        text: str,
        payload: Payload = None,
        semantic_keys: Optional[Sequence[SemanticKey]] = None,
        spatial_keys: Optional[Sequence[Sequence[float]]] = None,
        temporal_keys: Optional[Sequence[float]] = None,
        metadata: Optional[Metadata] = None,
        source_slot: Optional[str] = None,
    ) -> None:
        self.entry_id = entry_id
        self.text = text
        self.payload = payload
        # Defensive copies + normalisation to tuples for spatial keys.
        self.semantic_keys: List[SemanticKey] = (
            list(semantic_keys) if semantic_keys is not None else []
        )
        self.spatial_keys: List[SpatialKey] = (
            [tuple(float(c) for c in k) for k in spatial_keys]
            if spatial_keys is not None
            else []
        )
        self.temporal_keys: List[TemporalKey] = (
            [float(t) for t in temporal_keys] if temporal_keys is not None else []
        )
        self.metadata: Metadata = dict(metadata) if metadata else {}
        if source_slot is not None:
            self.source_slot = str(source_slot)
        else:
            self.source_slot = None

    # -- mutation helpers -------------------------------------------------- #
    def add_semantic(self, *keys: SemanticKey) -> "MemoryEntry":
        for k in keys:
            if k:
                self.semantic_keys.append(k)
        return self

    def add_spatial(self, *keys: Sequence[float]) -> "MemoryEntry":
        for k in keys:
            self.spatial_keys.append(tuple(float(c) for c in k))
        return self

    def add_temporal(self, *keys: float) -> "MemoryEntry":
        for k in keys:
            self.temporal_keys.append(float(k))
        return self

    # -- match predicates (exact; ranges handled by index/query) ---------- #
    def matches_semantic(self, key: SemanticKey) -> bool:
        return key in self.semantic_keys

    def matches_spatial(self, key: Sequence[float]) -> bool:
        target = tuple(float(c) for c in key)
        return target in self.spatial_keys

    def matches_temporal(self, key: float) -> bool:
        return float(key) in self.temporal_keys

    # -- dunder ------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MemoryEntry(id={self.entry_id!r}, text={self.text!r}, "
            f"sem={len(self.semantic_keys)}, spa={len(self.spatial_keys)}, "
            f"tem={len(self.temporal_keys)})"
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MemoryEntry) and other.entry_id == self.entry_id

    def __hash__(self) -> int:
        return hash(self.entry_id)


# --------------------------------------------------------------------------- #
# MultiAxisIndex  (utility, not part of any ABC)
# --------------------------------------------------------------------------- #
class MultiAxisIndex:
    """Inverted indices over the three MemoryEntry axes.

    Lookup is exact-match on semantic/spatial and range-match on temporal.
    The index does *not* own the entries (it stores ids only), so the caller
    keeps a separate ``id -> MemoryEntry`` map. This separation lets a module
    layer the index on top of whatever primary storage it prefers.
    """

    def __init__(self) -> None:
        self._sem: Dict[SemanticKey, Set[str]] = {}
        self._spa: Dict[SpatialKey, Set[str]] = {}
        self._tem: List[Tuple[float, str]] = []  # sorted on the fly when queried
        self._size = 0

    # -- mutation ---------------------------------------------------------- #
    def add(self, entry: MemoryEntry) -> None:
        """Insert (or update) an entry's keys in the index."""
        self.remove(entry.entry_id)  # idempotent re-index
        eid = entry.entry_id
        for k in entry.semantic_keys:
            self._sem.setdefault(k, set()).add(eid)
        for k in entry.spatial_keys:
            self._spa.setdefault(k, set()).add(eid)
        for t in entry.temporal_keys:
            self._tem.append((t, eid))
        self._size += 1

    def remove(self, entry_id: str) -> None:
        """Drop every key referring to ``entry_id``. Safe to call on absent ids."""
        for s in self._sem.values():
            s.discard(entry_id)
        self._sem = {k: v for k, v in self._sem.items() if v}
        for s in self._spa.values():
            s.discard(entry_id)
        self._spa = {k: v for k, v in self._spa.items() if v}
        before = len(self._tem)
        self._tem = [(t, e) for (t, e) in self._tem if e != entry_id]
        if before != len(self._tem):
            self._size -= 1

    def clear(self) -> None:
        self._sem.clear()
        self._spa.clear()
        self._tem.clear()
        self._size = 0

    # -- query ------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._size

    def lookup_semantic(self, key: SemanticKey) -> Set[str]:
        return set(self._sem.get(key, ()))

    def lookup_spatial(self, key: Sequence[float]) -> Set[str]:
        return set(self._spa.get(tuple(float(c) for c in key), ()))

    def lookup_temporal(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> Set[str]:
        """Return ids with at least one temporal key in ``[t_min, t_max]``.

        ``None`` bounds are treated as unbounded.
        """
        if t_min is None and t_max is None:
            return {eid for (_, eid) in self._tem}
        lo = float(t_min) if t_min is not None else float("-inf")
        hi = float(t_max) if t_max is not None else float("inf")
        return {eid for (t, eid) in self._tem if lo <= t <= hi}

    def lookup(
        self,
        semantic: Optional[Iterable[SemanticKey]] = None,
        spatial: Optional[Iterable[Sequence[float]]] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        require_any_axis: bool = False,
        within_axis_op: str = "intersect",
    ) -> Set[str]:
        """Multi-axis lookup.

        * Each provided axis produces a candidate id set; results are the
          *intersection* across axes (AND between axes).
        * Within an axis, multiple keys are combined by ``within_axis_op``:

          - ``"intersect"`` (default): AND — a query for ``["red","chair"]``
            returns only entries tagged with *both* "red" and "chair".
            Matches typical EQA multi-word query intent.
          - ``"union"``: OR — a query for ``["chair","sofa"]`` returns
            entries tagged with either. Useful for exploratory retrieval.
        * An axis with no key produced is treated as "no constraint".
        * If ``require_any_axis`` is True and no axis supplied any key, returns
          the empty set (instead of "all entries").
        """
        if within_axis_op not in ("intersect", "union"):
            raise ValueError(
                f"within_axis_op must be 'intersect' or 'union', got {within_axis_op!r}"
            )

        per_axis: List[Set[str]] = []
        if semantic is not None:
            sem_sets = [self.lookup_semantic(k) for k in semantic]
            per_axis.append(self._combine_within_axis(sem_sets, within_axis_op))
        if spatial is not None:
            spa_sets = [self.lookup_spatial(k) for k in spatial]
            per_axis.append(self._combine_within_axis(spa_sets, within_axis_op))
        if t_min is not None or t_max is not None:
            per_axis.append(self.lookup_temporal(t_min, t_max))

        if not per_axis:
            return set() if require_any_axis else {eid for (_, eid) in self._tem}

        # Intersect non-"no-match-but-supplied" sets.
        # An axis that was supplied but matched nothing still constrains the
        # result to nothing (proper AND semantics).
        result = per_axis[0]
        for s in per_axis[1:]:
            result &= s
        return result

    def all_ids(self) -> Set[str]:
        return {eid for (_, eid) in self._tem} | self._all_sem_spa_ids()

    def _all_sem_spa_ids(self) -> Set[str]:
        ids: Set[str] = set()
        for s in self._sem.values():
            ids |= s
        for s in self._spa.values():
            ids |= s
        return ids

    @staticmethod
    def _combine_within_axis(sets: List[Set[str]], op: str) -> Set[str]:
        """Combine per-key result sets within one axis."""
        # Skip "no constraint" markers (empty sets from absent keys are real
        # constraints, so we keep them; this only filters our own empties
        # produced when the caller passed no keys at all).
        if not sets:
            return set()
        if op == "union":
            combined: Set[str] = set()
            for s in sets:
                combined |= s
            return combined
        # intersect
        result = sets[0]
        for s in sets[1:]:
            result &= s
        return result


__all__ = ["MemoryEntry", "MultiAxisIndex"]
