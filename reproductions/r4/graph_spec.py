"""Wiring of the R4 memory module into a unimem :class:`MemoryGraph`.

R4 nominally has a single memory store (the 4D knowledge database D), but
the unimem graph lets us surround it with related slots to demonstrate the
*inter-slot* consistency checks the framework supports — useful for the
project's downstream memory-correction research:

* ``wm``  (WorkingMemoryABC)      — current question + last VLM output
* ``db``  (R4KnowledgeDatabase)   — the 4D database (slot=GM)
* ``sm``  (SemanticMemoryABC stub)— long-horizon facts distilled from D
* WM → DB (FEEDS): observations flow into the database
* DB → SM (CONSOLIDATES_TO): periodic fact distillation

The graph is built imperatively (rather than via ``GraphSpec``) because R4
needs runtime injection of the embedding fn / SLAM map — state that the
declarative spec can't easily capture. Production tools may prefer the
spec/registry route by stashing such deps in a closure-friendly container.
"""
from __future__ import annotations

from typing import Optional

from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slot_abc import SemanticMemoryABC, WorkingMemoryABC
from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind, MemoryEdge
from unimem.graph.graph import MemoryGraph
from unimem.graph.node import MemoryNode

from .memory.embedding import EmbeddingFn, get_default_embedding
from .memory.knowledge_db import R4KnowledgeDatabase
from .memory.slam_map import SimpleSLAMMap


# --------------------------------------------------------------------------- #
# Auxiliary stub modules (minimal WM + SM so the graph is meaningful)
# --------------------------------------------------------------------------- #
class _R4WorkingMemory(WorkingMemoryABC):
    def __init__(self):
        self._current = None

    def write(self, entry, context):
        self._current = entry
        return True

    def read(self, query):
        if self._current is None:
            return QueryResult()
        return QueryResult(entries=[self._current])

    def clear(self):
        self._current = None

    def stats(self):
        return {"count": 1 if self._current else 0}

    def get_current(self):
        return self._current

    def set_current(self, entry):
        self._current = entry


class _R4SemanticMemory(SemanticMemoryABC):
    """Trivial fact store; receives distilled facts via consolidation."""

    def __init__(self):
        self._facts = []  # list of (s, p, o)

    def write(self, entry, context):
        md = entry.metadata
        if "subject" in md:
            self._facts.append((md["subject"], md.get("predicate", "is"), md.get("object")))
            return True
        return False

    def read(self, query):
        entries = [
            MemoryEntry(
                f"f{i}",
                f"({s},{p},{o})",
                metadata={"subject": s, "predicate": p},
            )
            for i, (s, p, o) in enumerate(self._facts)
        ]
        return QueryResult(entries=entries)

    def clear(self):
        self._facts.clear()

    def stats(self):
        return {"count": len(self._facts)}

    def add_fact(self, s, p, o):
        self._facts.append((s, p, o))
        return True

    def query_facts(self, s=None, p=None, obj=None):
        out = []
        for (ss, pp, oo) in self._facts:
            if s is not None and ss != s:
                continue
            if p is not None and pp != p:
                continue
            if obj is not None and oo != obj:
                continue
            out.append((ss, pp, oo))
        return out


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_r4_graph(
    embedding_fn: Optional[EmbeddingFn] = None,
    slam_map: Optional[SimpleSLAMMap] = None,
) -> MemoryGraph:
    """Construct a 3-node unimem graph for R4 (WM → DB → SM)."""
    g = MemoryGraph()
    db = R4KnowledgeDatabase(embedding_fn=embedding_fn, slam_map=slam_map)
    g.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=_R4WorkingMemory()))
    g.add_node(MemoryNode(node_id="db", slot=MemorySlot.GM, module=db))
    g.add_node(MemoryNode(node_id="sm", slot=MemorySlot.SM, module=_R4SemanticMemory()))
    g.add_edge(MemoryEdge("wm", "db", EdgeKind.FEEDS))
    g.add_edge(MemoryEdge("db", "sm", EdgeKind.CONSOLIDATES_TO))
    return g


__all__ = ["build_r4_graph", "_R4WorkingMemory", "_R4SemanticMemory"]
