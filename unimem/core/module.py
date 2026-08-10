"""``MemoryModule`` — the core ABC that every concrete memory implements.

Design notes (see plan §"关键设计决策"):

* Only four *required* abstract methods: ``write``, ``read``, ``clear``,
  ``stats``. Everything else has a sensible default so a minimal slot
  implementation is not drowned in boilerplate.
* ``update`` (per-step maintenance) and ``consolidate`` (cross-module
  sedimentation) are hooks, not contract — modules that don't need them do
  nothing by default.
* Per-module policy slots (``write_policy`` / ``read_policy`` /
  ``forget_policy``) default to ``None`` which the graph layer interprets as
  "use the graph-level policy".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .context import MemoryContext
from .entry import MemoryEntry
from .query import Query, QueryResult


class MemoryModule(ABC):
    """Abstract base class for every concrete memory implementation."""

    # Per-module policies; ``None`` means "defer to graph-level policy".
    write_policy: Any = None
    read_policy: Any = None
    forget_policy: Any = None

    # ------------------------------------------------------------------ #
    # Required contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Attempt to ingest ``entry``.

        Returns ``True`` if anything was actually stored. Implementations
        should consult :attr:`write_policy` (if set) *before* mutating state.
        """

    @abstractmethod
    def read(self, query: Query) -> QueryResult:
        """Retrieve entries matching ``query``.

        Implementations should respect :attr:`read_policy` (if set) for
        post-processing such as concatenation, dedup, or reranking.
        """

    @abstractmethod
    def clear(self) -> None:
        """Drop every stored entry and reset all indices."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return a JSON-ish dict of diagnostic counters."""

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
        """Convenience entry-count helper.

        Default implementation reads ``stats()`` and looks for common keys
        (``count`` / ``entries`` / ``size`` / ``n``); falls back to ``-1`` if
        none are present, in which case subclasses should override.
        """
        s = self.stats()
        for k in ("count", "entries", "size", "n"):
            v = s.get(k)
            if isinstance(v, (int, float)) and v >= 0:
                return int(v)
        return -1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            return f"{type(self).__name__}(count={self.count()})"
        except Exception:
            return f"{type(self).__name__}()"


__all__ = ["MemoryModule"]
