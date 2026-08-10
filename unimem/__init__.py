"""``unimem`` — a slot-based, graph-organised memory framework for EQA agents.

Top-level design tenets (see ``research/research_notes.md`` for the survey
that motivated them):

* Six *functional slots* (WM / SG / GM / EM / SM / PM) — every observed EQA
  memory module fits one.
* Modules are *nodes* in a typed graph; data flow, sedimentation, indexing,
  cross-references, and hierarchy are *edges*.
* The framework defines **interface contracts only**; concrete implementations
  plug in their own storage and algorithms.
* Pure stdlib, Python 3.9+.

Quick start::

    from unimem import (
        MemorySlot, MemoryEntry, MemoryContext, Query, QueryBuilder,
        MemoryGraph, MemoryNode, MemoryEdge, EdgeKind,
        Registry, MemoryGraphBuilder, GraphSpec, NodeSpec, EdgeSpec,
        ListEpisodicMemory, FIFOForgetPolicy, ExtractFactsConsolidationPolicy,
    )
"""
from __future__ import annotations

# Core data types
from .core.context import MemoryContext
from .core.entry import MemoryEntry, MultiAxisIndex
from .core.module import MemoryModule
from .core.query import Query, QueryBuilder, QueryResult
from .core.slot_abc import (
    EpisodicMemoryABC,
    ProceduralMemoryABC,
    SceneGraphMemoryABC,
    SemanticMemoryABC,
    SpatialGeometricMemoryABC,
    WorkingMemoryABC,
)
from .core.slots import MemorySlot

# Graph
from .graph.builder import EdgeSpec, GraphSpec, MemoryGraphBuilder, NodeSpec
from .graph.edge import EdgeKind, MemoryEdge
from .graph.graph import MemoryGraph
from .graph.node import MemoryNode

# Factory
from .factory.memory_factory import MemoryFactory
from .factory.registry import Registry

# Policies
from .policies.consolidation_policy import ConsolidationPolicy, Passthrough
from .policies.forget_policy import ForgetPolicy, NoOp
from .policies.read_policy import ConcatRead, FirstNonEmptyRead, ReadPolicy
from .policies.write_policy import (
    AlwaysWrite,
    LambdaWritePolicy,
    NeverWrite,
    WritePolicy,
)

# Reference implementations
from .reference.consolidate_extract import ExtractFactsConsolidationPolicy
from .reference.episodic_memory import ListEpisodicMemory
from .reference.forget_fifo import FIFOForgetPolicy

__version__ = "0.1.0"

__all__ = [
    # core
    "MemorySlot",
    "MemoryEntry",
    "MultiAxisIndex",
    "MemoryContext",
    "MemoryModule",
    "Query",
    "QueryBuilder",
    "QueryResult",
    "WorkingMemoryABC",
    "SceneGraphMemoryABC",
    "SpatialGeometricMemoryABC",
    "EpisodicMemoryABC",
    "SemanticMemoryABC",
    "ProceduralMemoryABC",
    # graph
    "MemoryNode",
    "MemoryEdge",
    "EdgeKind",
    "MemoryGraph",
    "GraphSpec",
    "NodeSpec",
    "EdgeSpec",
    "MemoryGraphBuilder",
    # factory
    "Registry",
    "MemoryFactory",
    # policies
    "WritePolicy",
    "AlwaysWrite",
    "NeverWrite",
    "LambdaWritePolicy",
    "ReadPolicy",
    "ConcatRead",
    "FirstNonEmptyRead",
    "ConsolidationPolicy",
    "Passthrough",
    "ForgetPolicy",
    "NoOp",
    # reference
    "ListEpisodicMemory",
    "FIFOForgetPolicy",
    "ExtractFactsConsolidationPolicy",
    # meta
    "__version__",
]
