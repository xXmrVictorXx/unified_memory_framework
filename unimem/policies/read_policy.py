"""Read-side merge policies.

A ``ReadPolicy`` runs after the graph has collected one
:class:`~unimem.core.query.QueryResult` per node. It can merge, dedup, rerank,
or simply pass through. Default ``ConcatRead`` flattens everything into a
single result, preserving per-entry provenance via the underlying result
metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.entry import MemoryEntry
from ..core.query import QueryResult


class ReadPolicy(ABC):
    """Combine multiple per-node ``QueryResult``s into one."""

    @abstractmethod
    def merge(self, results: List[QueryResult]) -> QueryResult:
        """Return a single merged result."""


class ConcatRead(ReadPolicy):
    """Concatenate entries from every result in input order.

    Score lists are dropped (cannot be meaningfully aligned across modules).
    The returned result's ``metadata`` records the per-source counts.
    """

    def merge(self, results: List[QueryResult]) -> QueryResult:
        merged = QueryResult()
        counts: dict = {}
        for r in results:
            if not r.entries:
                continue
            merged.entries.extend(r.entries)
            key = r.source_node_id or r.source_slot or "?"
            counts[key] = counts.get(key, 0) + len(r.entries)
        merged.scores = None  # cross-module scores are not comparable
        merged.metadata["per_source_counts"] = counts
        merged.metadata["n_sources"] = len(counts)
        return merged


class FirstNonEmptyRead(ReadPolicy):
    """Return the first non-empty result; useful for short-circuiting planners."""

    def merge(self, results: List[QueryResult]) -> QueryResult:
        for r in results:
            if r.entries:
                return r
        return QueryResult.empty()


__all__ = ["ReadPolicy", "ConcatRead", "FirstNonEmptyRead"]
