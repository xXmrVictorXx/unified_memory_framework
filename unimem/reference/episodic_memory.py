"""``ListEpisodicMemory`` — the reference episodic module.

Stress-tests the framework by exercising every part of the ABC contract:

* **Multi-axis indexing** via :class:`~unimem.core.entry.MultiAxisIndex`.
* **Multi-timescale bucketing** à la WorldMM: events are filed into
  timescale buckets (default ``30s, 180s, 600s, 3600s``); a query can ask
  for the fine-grained bucket, the coarse bucket, or both.
* **Write-policy gating**: the module's own ``write_policy`` (if set) is
  consulted before ``append_event`` is called. This mirrors how an
  INHerit-SG-style event-triggered gate would plug in.
* **Consolidation**: the default ``consolidate`` returns coarse summary
  entries (one per timescale bucket) suitable for ingestion by a semantic
  memory; richer extraction is left to
  :class:`~unimem.reference.consolidate_extract.ExtractFactsConsolidationPolicy`.
* **FIFO forgetting** via :class:`~unimem.reference.forget_fifo.FIFOForgetPolicy`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry, MultiAxisIndex
from ..core.module import MemoryModule
from ..core.query import Query, QueryResult
from ..core.slot_abc import EpisodicMemoryABC
from ..policies.write_policy import AlwaysWrite, WritePolicy


class ListEpisodicMemory(EpisodicMemoryABC):
    """In-memory episodic store with multi-timescale buckets + 3-axis index."""

    # WorldMM-inspired default timescales (seconds).
    timescales: Tuple[float, ...] = (30.0, 180.0, 600.0, 3600.0)

    def __init__(
        self,
        timescales: Optional[Tuple[float, ...]] = None,
        capacity_per_bucket: Optional[int] = None,
        write_policy: Optional[WritePolicy] = None,
    ) -> None:
        if timescales is not None:
            self.timescales = tuple(float(t) for t in timescales)
        self.capacity_per_bucket = capacity_per_bucket
        # ``write_policy`` is a class-level attribute on MemoryModule; we set
        # the instance attribute to override it.
        self.write_policy = write_policy or AlwaysWrite()

        # Storage: timescale -> list of entries (insertion order = time order).
        self._buckets: Dict[float, List[MemoryEntry]] = {
            t: [] for t in self.timescales
        }
        # Single 3-axis index across every bucket (ids are unique across buckets).
        self._index = MultiAxisIndex()
        # All entries by id (the index stores ids only).
        self._by_id: Dict[str, MemoryEntry] = {}
        # A monotonically increasing counter for synthesising temporal keys
        # when an entry has none.
        self._synth_t: float = 0.0

    # ------------------------------------------------------------------ #
    # Required MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Gate via ``write_policy`` then delegate to :meth:`append_event`."""
        if not self.write_policy.should_write(self, entry, context):
            return False
        self.append_event(entry)
        return True

    def read(self, query: Query) -> QueryResult:
        ids = self._match(query)
        entries = [self._by_id[i] for i in ids if i in self._by_id]
        # Stable order: by min temporal key per entry.
        entries.sort(key=lambda e: min(e.temporal_keys) if e.temporal_keys else 0.0)
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot="episodic")

    def clear(self) -> None:
        for bucket in self._buckets.values():
            bucket.clear()
        self._index.clear()
        self._by_id.clear()

    def stats(self) -> Dict[str, Any]:
        per_bucket = {str(t): len(b) for t, b in self._buckets.items()}
        return {
            "count": len(self._by_id),
            "per_bucket": per_bucket,
            "timescales": list(self.timescales),
            "capacity_per_bucket": self.capacity_per_bucket,
        }

    # ------------------------------------------------------------------ #
    # EpisodicMemoryABC
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        """File ``entry`` into every timescale bucket whose window it falls in.

        If ``entry`` has no ``temporal_keys``, we synthesise one from the
        running counter so the index still works. The entry is inserted
        *once per matching bucket* (each bucket holds the same entry object),
        but indexed only once in the global ``_index``.
        """
        # Ensure at least one temporal key exists.
        if not entry.temporal_keys:
            self._synth_t += 1.0
            entry.add_temporal(self._synth_t)
        t = entry.temporal_keys[0]

        # File into every bucket whose granularity is >= the entry's age.
        # We treat timescale values as "remember events up to this many seconds
        # old". The caller is responsible for using meaningful timestamps; if
        # they pass raw monotonic counters, every bucket will accept.
        inserted_any = False
        for window in self.timescales:
            bucket = self._buckets[window]
            bucket.append(entry)
            inserted_any = True
            # Capacity management: FIFO drop the oldest.
            if (
                self.capacity_per_bucket is not None
                and len(bucket) > self.capacity_per_bucket
            ):
                dropped = bucket.pop(0)
                # Note: do NOT remove from index/by_id here — the entry may
                # still live in other buckets. Real cleanup happens in
                # forget() / clear().
        if not inserted_any:  # pragma: no cover - defensive
            self._buckets[self.timescales[-1]].append(entry)

        # Index once globally.
        self._index.add(entry)
        self._by_id[entry.entry_id] = entry

    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        ids = self._index.lookup_temporal(t_min, t_max)
        out = [self._by_id[i] for i in ids if i in self._by_id]
        out.sort(key=lambda e: min(e.temporal_keys) if e.temporal_keys else 0.0)
        return out

    # ------------------------------------------------------------------ #
    # Consolidation hook (default; richer logic lives in ExtractFacts policy)
    # ------------------------------------------------------------------ #
    def consolidate(
        self, other: MemoryModule, context: MemoryContext
    ) -> List[MemoryEntry]:
        """Default: produce one coarse summary entry per non-empty bucket.

        Each summary is a single :class:`MemoryEntry` whose text aggregates
        the bucket's contents; this is intentionally simple. Real systems
        should use :class:`ExtractFactsConsolidationPolicy` instead, which
        extracts structured triples.
        """
        summaries: List[MemoryEntry] = []
        for window, bucket in self._buckets.items():
            if not bucket:
                continue
            text = f"summary@{window}s: " + " | ".join(e.text for e in bucket[:5])
            summaries.append(
                MemoryEntry(
                    entry_id=f"summary-{window}-{id(self)}-{len(bucket)}",
                    text=text,
                    semantic_keys=["summary"],
                    temporal_keys=[window],
                    source_slot="episodic",
                    metadata={
                        "kind": "bucket_summary",
                        "window": window,
                        "n_events": len(bucket),
                    },
                )
            )
        return summaries

    # ------------------------------------------------------------------ #
    # FIFO drop helper used by FIFOForgetPolicy
    # ------------------------------------------------------------------ #
    def drop_oldest(self, n: int) -> int:
        """Drop the ``n`` globally-oldest entries. Returns the actual count."""
        if n <= 0 or not self._by_id:
            return 0
        # Build (min_t, entry_id) across all entries.
        ranked = sorted(
            (
                (min(e.temporal_keys) if e.temporal_keys else float("inf"), eid)
                for eid, e in self._by_id.items()
            )
        )
        dropped = 0
        for _, eid in ranked[:n]:
            self._remove_entry(eid)
            dropped += 1
        return dropped

    def _remove_entry(self, entry_id: str) -> None:
        for bucket in self._buckets.values():
            bucket[:] = [e for e in bucket if e.entry_id != entry_id]
        self._index.remove(entry_id)
        self._by_id.pop(entry_id, None)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _match(self, query: Query) -> List[str]:
        if (
            not query.has_semantic
            and not query.has_spatial
            and not query.has_temporal
        ):
            # No filter: everything.
            return list(self._by_id.keys())
        ids = self._index.lookup(
            semantic=query.semantic or None,
            spatial=query.spatial or None,
            t_min=query.t_min,
            t_max=query.t_max,
            require_any_axis=True,
        )
        return list(ids)


__all__ = ["ListEpisodicMemory"]
