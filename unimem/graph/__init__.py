"""Memory graph components.

The graph is the heart of the framework. Nodes are memory modules; edges are
typed relations (data flow, consolidation, indexing, references, hierarchy).
The three core graph algorithms are:

1. **Fan-out read** — broadcast a query to every (slot-filtered) node.
2. **Fan-in write** — write an entry then propagate along FEEDS edges
   (BFS, with optional per-edge ``WritePolicy`` gating).
3. **Consolidation pass** — traverse CONSOLIDATES_TO edges to extract
   sedimentable entries from each source into its target, then run the
   forget policy on every node.
"""
from __future__ import annotations

from .builder import (
    EdgeSpec,
    GraphSpec,
    MemoryGraphBuilder,
    NodeSpec,
)
from .edge import EdgeKind, MemoryEdge
from .graph import MemoryGraph
from .node import MemoryNode

__all__ = [
    "EdgeKind",
    "MemoryEdge",
    "MemoryNode",
    "MemoryGraph",
    "GraphSpec",
    "NodeSpec",
    "EdgeSpec",
    "MemoryGraphBuilder",
]
