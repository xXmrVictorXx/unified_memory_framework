"""``unimem`` — a slot-based, graph-organised memory framework for EQA agents.

Top-level design tenets (see ``research/research_notes.md`` for the survey
that motivated them):

* Six *functional slots* (WM / SG / GM / EM / SM / PM) — every observed EQA
  memory module fits one.
* Modules are *nodes* in a typed graph; data flow, sedimentation, indexing,
  cross-references, and hierarchy are *edges*.
* The framework defines **interface contracts only**; concrete implementations
  plug in their own storage and algorithms.
* Pluggable storage backends (in-memory, Neo4j, Qdrant) via
  :mod:`unimem.graph_storage` + :mod:`unimem.vector_storage`.

Quick start::

    from unimem import (
        MemorySlot, MemoryEntry, MemoryContext, Query, QueryBuilder,
        MemoryGraph, MemoryNode, MemoryEdge, EdgeKind,
        Registry, MemoryGraphBuilder, GraphSpec, NodeSpec, EdgeSpec,
        ListEpisodicMemory, FIFOForgetPolicy, ExtractFactsConsolidationPolicy,
        create_graph_storage, create_vector_storage, DedupPolicy,
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
    DedupPolicy,
    LambdaWritePolicy,
    NeverWrite,
    WritePolicy,
)

# Reference implementations
from .reference.consolidate_extract import ExtractFactsConsolidationPolicy
from .reference.episodic_memory import ListEpisodicMemory
from .reference.forget_fifo import FIFOForgetPolicy

# Storage backends
from .graph_storage import (
    GraphStorage,
    InMemoryGraphStorage,
    create_graph_storage,
)
from .vector_storage import (
    InMemoryVectorStorage,
    VectorStorage,
    create_vector_storage,
)

# Op log
from .op_log import OpLog, OpLogEntry, SQLiteOpLog

# Config
from .config import (
    EdgeSpec as ConfigEdgeSpec,  # noqa: F401 — re-export alias
)
from .config import (
    ModuleSpec,
    StorageConfig,
    UnimemConfig,
    build_graph,
    build_storage,
    load_config,
    load_unimem,
)

__version__ = "0.2.0"

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
    "DedupPolicy",
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
    # storage
    "GraphStorage",
    "InMemoryGraphStorage",
    "create_graph_storage",
    "VectorStorage",
    "InMemoryVectorStorage",
    "create_vector_storage",
    # op log
    "OpLog",
    "OpLogEntry",
    "SQLiteOpLog",
    # config
    "ModuleSpec",
    "StorageConfig",
    "UnimemConfig",
    "build_graph",
    "build_storage",
    "load_config",
    "load_unimem",
    # meta
    "__version__",
]
