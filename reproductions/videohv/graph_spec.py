"""Wire VideoHV-Agent's two synthesised memories into a unimem graph.

Topology:

* ``summary`` (VideoSummaryMemory) — episodic store of clip summaries
* ``trace``   (VerificationTraceMemory) — short-term round-by-round log
* trace → summary (REFERENCES): every verification round can read the
  clip summaries. We use REFERENCES rather than FEEDS because the
  verification *reads* summaries but doesn't transform them.
"""
from __future__ import annotations

from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind, MemoryEdge
from unimem.graph.graph import MemoryGraph
from unimem.graph.node import MemoryNode

from .memory.time_verification_trace import VerificationTraceMemory
from .memory.video_summary_memory import VideoSummaryMemory


def build_videohv_graph(
    summary_memory: VideoSummaryMemory = None,
    trace_memory: VerificationTraceMemory = None,
) -> MemoryGraph:
    g = MemoryGraph()
    g.add_node(MemoryNode(
        node_id="summary",
        slot=MemorySlot.EM,
        module=summary_memory or VideoSummaryMemory(),
    ))
    g.add_node(MemoryNode(
        node_id="trace",
        slot=MemorySlot.EM,
        module=trace_memory or VerificationTraceMemory(),
    ))
    g.add_edge(MemoryEdge("trace", "summary", EdgeKind.REFERENCES))
    return g


__all__ = ["build_videohv_graph"]
