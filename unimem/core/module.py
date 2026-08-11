"""``MemoryModule`` — the core ABC that every concrete memory implements.

Design notes (see plan §"关键设计决策"):

* The four historically-abstract methods (``write``, ``read``, ``clear``,
  ``stats``) now have **default implementations** that route through a
  :class:`~unimem.graph_storage.base.GraphStorage` instance. Concrete
  subclasses can still override any of them to provide bespoke behaviour
  (e.g. multi-timescale bucketing, custom indices, ...).
* ``graph_storage`` is an optional constructor argument. When omitted, the
  module behaves like the legacy in-memory implementations — this keeps
  every existing test passing while letting new code opt into storage
  persistence.
* ``update`` (per-step maintenance) and ``consolidate`` (cross-module
  sedimentation) are hooks, not contract — modules that don't need them do
  nothing by default.
* Per-module policy slots (``write_policy`` / ``read_policy`` /
  ``forget_policy``) default to ``None`` which the graph layer interprets as
  "use the graph-level policy".
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .context import MemoryContext
from .entry import MemoryEntry
from .query import Query, QueryResult
from .slots import MemorySlot

if TYPE_CHECKING:
    from ..graph_storage.base import GraphStorage
    from ..policies.forget_policy import ForgetPolicy
    from ..policies.read_policy import ReadPolicy
    from ..policies.write_policy import WritePolicy


class MemoryModule:
    """Base class for every concrete memory implementation.

    Construction
    ------------
    slot:
        The :class:`MemorySlot` this module occupies.
    graph_storage:
        Optional :class:`GraphStorage`. When provided, the default
        ``write`` / ``read`` / ``clear`` / ``stats`` implementations route
        through it — no subclassing needed. When omitted, the methods raise
        on call (subclasses are expected to override them with in-memory
        behaviour).
    write_policy / read_policy / forget_policy:
        Optional per-module policies. ``None`` means "defer to the
        graph-level policy".
    """

    # Per-module policies; ``None`` means "defer to graph-level policy".
    # These are instance attributes set in __init__.
    write_policy: Optional["WritePolicy"] = None
    read_policy: Optional["ReadPolicy"] = None
    forget_policy: Optional["ForgetPolicy"] = None

    def __init__(
        self,
        slot: MemorySlot,
        graph_storage: Optional["GraphStorage"] = None,
        write_policy: Optional["WritePolicy"] = None,
        read_policy: Optional["ReadPolicy"] = None,
        forget_policy: Optional["ForgetPolicy"] = None,
        *,
        impl_name: str = "default",
    ) -> None:
        if not isinstance(slot, MemorySlot):
            slot = MemorySlot.from_value(slot)
        self._slot = slot
        self._graph_storage = graph_storage
        self._impl_name = impl_name
        # Set per-instance policies (defaults remain None).
        if write_policy is not None:
            self.write_policy = write_policy
        if read_policy is not None:
            self.read_policy = read_policy
        if forget_policy is not None:
            self.forget_policy = forget_policy

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def slot(self) -> MemorySlot:
        return self._slot

    @property
    def graph_storage(self) -> Optional["GraphStorage"]:
        return self._graph_storage

    @property
    def impl_name(self) -> str:
        return self._impl_name

    # ------------------------------------------------------------------ #
    # Storage-backed default contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Default storage-backed write.

        1. Consult ``write_policy`` if set; reject if it returns False.
        2. Persist via ``graph_storage.add_memory_node``.
        """
        if self._graph_storage is None:
            raise RuntimeError(
                f"{type(self).__name__}.write() requires graph_storage — "
                "either pass one to __init__ or override write()."
            )
        if self.write_policy is not None and not self.write_policy.should_write(
            self, entry, context
        ):
            return False
        return self._graph_storage.add_memory_node(self._slot, entry)

    def read(self, query: Query) -> QueryResult:
        """Default storage-backed read."""
        if self._graph_storage is None:
            raise RuntimeError(
                f"{type(self).__name__}.read() requires graph_storage — "
                "either pass one to __init__ or override read()."
            )
        time_range = None
        if query.t_min is not None or query.t_max is not None:
            time_range = (query.t_min, query.t_max)
        entries = self._graph_storage.query_memories(
            slot=self._slot,
            semantic_keys=query.semantic or None,
            time_range=time_range,
        )
        if query.top_k is not None:
            entries = entries[: query.top_k]
        result = QueryResult(entries=entries, source_slot=self._slot.value)
        # Apply read_policy post-processing if set
        if self.read_policy is not None:
            result = self.read_policy.merge([result])
        return result

    def clear(self) -> None:
        """Default: delete every node labelled with this slot."""
        if self._graph_storage is None:
            raise RuntimeError(
                f"{type(self).__name__}.clear() requires graph_storage — "
                "either pass one to __init__ or override clear()."
            )
        self._graph_storage.query(
            f"MATCH (n:{self._slot.value}) DETACH DELETE n", {}
        )

    def stats(self) -> Dict[str, Any]:
        if self._graph_storage is None:
            return {"slot": self._slot.value, "count": 0, "impl": self._impl_name}
        rows = self._graph_storage.query(
            f"MATCH (n:{self._slot.value}) RETURN COUNT(*)"
        )
        count = rows[0].get("count", 0) if rows else 0
        return {
            "slot": self._slot.value,
            "impl": self._impl_name,
            "count": int(count),
        }

    # ------------------------------------------------------------------ #
    # Default-implemented hooks
    # ------------------------------------------------------------------ #
    def update(self, context: MemoryContext) -> None:
        """Per-step maintenance hook (decay, cache refresh, ...).

        Default: no-op. Override to implement decay, working-memory
        displacement, periodic re-indexing, etc.
        """
        return None

    def consolidate(
        self, other: "MemoryModule", context: MemoryContext
    ) -> List[MemoryEntry]:
        """Extract sedimentable entries from ``other`` into ``self``.

        Default: nothing. Override to implement episodic→semantic style
        consolidation. The graph layer's consolidation pass also lets you
        supply a per-edge :class:`ConsolidationPolicy` instead, in which
        case it bypasses this method.
        """
        return []

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """Convenience entry-count helper."""
        s = self.stats()
        for k in ("count", "entries", "size", "n"):
            v = s.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                return int(v)
        return -1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            return f"{type(self).__name__}(slot={self._slot.name}, count={self.count()})"
        except Exception:
            return f"{type(self).__name__}(slot={self._slot.name})"


__all__ = ["MemoryModule"]
