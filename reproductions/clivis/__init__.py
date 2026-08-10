"""CLiViS reproduction — three memory modules + iterative pipeline.

Reproduces CLiViS (CVPR 2026): a training-free EVR (Embodied Visual
Reasoning) framework that maintains a *Cognitive Map* across egocentric
video segments and iterates between LLM planning and VLM perception.

Original implementation lives at ``reproduce/CLiViS/`` (read-only reference).
This package re-implements its memory subsystem as three unimem
``MemoryModule`` subclasses, using pure Python instead of Neo4j so the
reproduction runs without external services.

Memory mapping (paper → unimem):

* ``TimeWorkingMemory`` (Rationale list + chat history) → :class:`TimeWorkingMemory`
  implements both :class:`~unimem.core.slot_abc.WorkingMemoryABC` and
  :class:`~unimem.core.slot_abc.EpisodicMemoryABC`.
* ``NavigationGraph`` (period → areas/objects/activities) → :class:`NavigationGraph`
  implements :class:`~unimem.core.slot_abc.SpatialGeometricMemoryABC`.
* ``RelationGraph`` (Neo4j property graph) → :class:`RelationGraph`
  implements :class:`~unimem.core.slot_abc.SceneGraphMemoryABC` and
  :class:`~unimem.core.slot_abc.SemanticMemoryABC`.
"""
from __future__ import annotations

from .memory.navigation_graph import NavigationGraph
from .memory.relation_graph import NodeLabels, RelationGraph
from .memory.time_working_memory import Rationale, TimeWorkingMemory
from .pipeline import CLiViSPipeline, CLiViSResult

__all__ = [
    "Rationale",
    "TimeWorkingMemory",
    "NavigationGraph",
    "RelationGraph",
    "NodeLabels",
    "CLiViSPipeline",
    "CLiViSResult",
]
