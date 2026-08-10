"""Wire CLiViS's three memory modules into a unimem :class:`MemoryGraph`.

Topology mirrors the original CLiViS pipeline's data flow:

* WM (TimeWorkingMemory) — short-term rationales & chat history
* SG (RelationGraph)     — Neo4j-style scene graph
* GM (NavigationGraph)   — period → entities index
* WM → SG (FEEDS)        — VLM responses can refine the scene graph
* WM → GM (FEEDS)        — VLM responses can register new objects in periods
* SG → SG (REFERENCES)   — not used here; placeholder
* SG → GM (INDEXES)      — scene-graph nodes index periods via area/period tags
"""
from __future__ import annotations

from typing import Optional

from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind, MemoryEdge
from unimem.graph.graph import MemoryGraph
from unimem.graph.node import MemoryNode

from .memory.navigation_graph import NavigationGraph
from .memory.relation_graph import RelationGraph
from .memory.time_working_memory import TimeWorkingMemory


def build_clivis_graph(
    question: str = "",
    periods: Optional[dict] = None,
) -> MemoryGraph:
    """Build the 3-node CLiViS-style unimem graph."""
    g = MemoryGraph()
    wm = TimeWorkingMemory(question=question)
    sg = RelationGraph()
    gm = NavigationGraph(period_description_dict=periods or {})
    g.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=wm))
    g.add_node(MemoryNode(node_id="sg", slot=MemorySlot.SG, module=sg))
    g.add_node(MemoryNode(node_id="gm", slot=MemorySlot.GM, module=gm))
    g.add_edge(MemoryEdge("wm", "sg", EdgeKind.FEEDS))
    g.add_edge(MemoryEdge("wm", "gm", EdgeKind.FEEDS))
    g.add_edge(MemoryEdge("sg", "gm", EdgeKind.INDEXES))
    return g


__all__ = ["build_clivis_graph"]
