"""``unimem.config`` — declarative graph + storage configuration.

Two equivalent entry points:

* **Code-driven** — :func:`build_graph` constructs a
  :class:`~unimem.graph.graph.MemoryGraph` from in-memory objects
  (``GraphStorage``, ``VectorStorage``, ``OpLog``, module/edge specs).
* **YAML-driven** — :func:`load_config` parses a YAML file into a
  :class:`UnimemConfig`; :func:`load_unimem` does the same and immediately
  builds the graph.

The YAML path is useful when reproductions want to version their graph
topology alongside the code; the code path is better for tests and
ad-hoc scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .core.module import MemoryModule
from .core.slots import MemorySlot
from .graph.edge import EdgeKind
from .graph.graph import MemoryGraph
from .graph.node import MemoryNode
from .graph_storage import GraphStorage, create_graph_storage
from .op_log import OpLog, SQLiteOpLog
from .vector_storage import VectorStorage, create_vector_storage


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class StorageConfig:
    """Storage backend configuration."""

    graph: Dict[str, Any] = field(default_factory=lambda: {"backend": "memory"})
    vector: Dict[str, Any] = field(default_factory=lambda: {"backend": "memory"})
    op_log: Optional[Dict[str, Any]] = None  # None = disabled


@dataclass
class ModuleSpec:
    """One node in the module topology."""

    node_id: str
    slot: str
    impl: str = "default"
    label: str = ""
    write_policy: Optional[Dict[str, Any]] = None
    read_policy: Optional[Dict[str, Any]] = None
    forget_policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """One edge in the module topology."""

    source: str
    target: str
    kind: str = "FEEDS"
    policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnimemConfig:
    """Top-level config: storage + module topology."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    modules: List[ModuleSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)
    default_write_policy: Optional[Dict[str, Any]] = None
    default_read_policy: Optional[Dict[str, Any]] = None
    default_forget_policy: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UnimemConfig":
        """Build from a parsed dict (e.g. ``yaml.safe_load``)."""
        storage_raw = raw.get("storage") or {}
        storage = StorageConfig(
            graph=dict(storage_raw.get("graph") or {"backend": "memory"}),
            vector=dict(storage_raw.get("vector") or {"backend": "memory"}),
            op_log=dict(storage_raw["op_log"]) if storage_raw.get("op_log") else None,
        )
        modules = [ModuleSpec(**m) for m in (raw.get("modules") or [])]
        edges = [EdgeSpec(**e) for e in (raw.get("edges") or [])]
        return cls(
            storage=storage,
            modules=modules,
            edges=edges,
            default_write_policy=raw.get("default_write_policy"),
            default_read_policy=raw.get("default_read_policy"),
            default_forget_policy=raw.get("default_forget_policy"),
        )


# --------------------------------------------------------------------------- #
# YAML loader
# --------------------------------------------------------------------------- #
def load_config(yaml_path: str) -> UnimemConfig:
    """Parse ``yaml_path`` into a :class:`UnimemConfig`.

    Requires PyYAML. Callers without PyYAML can construct UnimemConfig
    directly via :meth:`UnimemConfig.from_dict` with a parsed dict from
    any other source.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "load_config requires PyYAML; install via `pip install pyyaml`"
        ) from exc
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping; got {type(raw).__name__}")
    return UnimemConfig.from_dict(raw)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_storage(
    config: StorageConfig,
) -> tuple:
    """Build the (graph_storage, vector_storage, op_log) triple."""
    gs = create_graph_storage(config.graph)
    vs = create_vector_storage(config.vector)
    op_log: Optional[OpLog] = None
    if config.op_log and config.op_log.get("enabled", True):
        path = config.op_log.get("path")
        op_log = SQLiteOpLog(path=path)
    return gs, vs, op_log


def _resolve_policy(
    spec: Optional[Dict[str, Any]],
) -> Optional[Any]:
    """Instantiate a policy from a ``{"type": "...", ...}`` dict."""
    if spec is None:
        return None
    spec = dict(spec)
    ptype = spec.pop("type", None)
    if ptype is None:
        return None
    # Local import to avoid hard policy dependencies in this module.
    if ptype == "always":
        from .policies.write_policy import AlwaysWrite
        return AlwaysWrite()
    if ptype == "never":
        from .policies.write_policy import NeverWrite
        return NeverWrite()
    if ptype == "concat":
        from .policies.read_policy import ConcatRead
        return ConcatRead()
    if ptype == "noop":
        from .policies.forget_policy import NoOp
        return NoOp()
    if ptype == "fifo":
        from .reference.forget_fifo import FIFOForgetPolicy
        return FIFOForgetPolicy(capacity=spec.get("capacity"))
    if ptype == "dedup":
        # DedupPolicy requires storages; defer to caller — return None here
        # so callers that need DedupPolicy can construct it explicitly.
        return None
    return None


def build_graph(
    graph_storage: Optional[GraphStorage] = None,
    vector_storage: Optional[VectorStorage] = None,
    op_log: Optional[OpLog] = None,
    modules_spec: Optional[List[ModuleSpec]] = None,
    edges_spec: Optional[List[EdgeSpec]] = None,
    default_write_policy: Optional[Any] = None,
    default_read_policy: Optional[Any] = None,
    default_forget_policy: Optional[Any] = None,
) -> MemoryGraph:
    """Code-driven :class:`MemoryGraph` construction.

    Each ``ModuleSpec`` becomes a :class:`MemoryNode` whose
    :class:`MemoryModule` is built with the slot's default storage-backed
    implementation (i.e. ``MemoryModule(slot=..., graph_storage=...)``).
    """
    # Local import to allow the function to be referenced before MemoryGraph
    # gains its storage-aware constructor in Phase 4.
    from .graph.edge import MemoryEdge

    graph_storage = graph_storage or create_graph_storage({"backend": "memory"})
    modules_spec = modules_spec or []
    edges_spec = edges_spec or []

    graph_kwargs: Dict[str, Any] = {}
    # Pass storage only if MemoryGraph's __init__ accepts it (Phase 4 onward).
    init_params = _memory_graph_init_params()
    if "graph_storage" in init_params:
        graph_kwargs["graph_storage"] = graph_storage
    if "op_log" in init_params:
        graph_kwargs["op_log"] = op_log
    if "default_write_policy" in init_params and default_write_policy is not None:
        graph_kwargs["default_write_policy"] = default_write_policy
    if "default_read_policy" in init_params and default_read_policy is not None:
        graph_kwargs["default_read_policy"] = default_read_policy
    if "default_forget_policy" in init_params and default_forget_policy is not None:
        graph_kwargs["default_forget_policy"] = default_forget_policy

    graph = MemoryGraph(**graph_kwargs)

    # Materialise modules
    for spec in modules_spec:
        slot = MemorySlot.from_value(spec.slot)
        module_kwargs: Dict[str, Any] = {"slot": slot}
        module_init = _memory_module_init_params()
        if "graph_storage" in module_init:
            module_kwargs["graph_storage"] = graph_storage
        module = MemoryModule(**module_kwargs)
        if spec.write_policy:
            module.write_policy = _resolve_policy(spec.write_policy)
        if spec.read_policy:
            module.read_policy = _resolve_policy(spec.read_policy)
        if spec.forget_policy:
            module.forget_policy = _resolve_policy(spec.forget_policy)
        node = MemoryNode(
            node_id=spec.node_id,
            slot=slot,
            module=module,
            label=spec.label,
            metadata=dict(spec.metadata),
        )
        graph.add_node(node)

    # Materialise edges
    for spec in edges_spec:
        kind = EdgeKind.from_value(spec.kind)
        edge = MemoryEdge(
            source_id=spec.source,
            target_id=spec.target,
            kind=kind,
            metadata=dict(spec.metadata),
        )
        graph.add_edge(edge)

    return graph


def _memory_graph_init_params():
    import inspect
    sig = inspect.signature(MemoryGraph.__init__)
    return set(sig.parameters.keys())


def _memory_module_init_params():
    import inspect
    sig = inspect.signature(MemoryModule.__init__)
    return set(sig.parameters.keys())


def load_unimem(yaml_path: str) -> MemoryGraph:
    """Parse YAML + build a fully-wired :class:`MemoryGraph`.

    For policies that require runtime dependencies (e.g. ``DedupPolicy``
    which needs an embedding function), build the graph manually via
    :func:`build_graph` and attach the policy to the module afterwards.
    """
    cfg = load_config(yaml_path)
    gs, vs, op_log = build_storage(cfg.storage)
    return build_graph(
        graph_storage=gs,
        vector_storage=vs,
        op_log=op_log,
        modules_spec=cfg.modules,
        edges_spec=cfg.edges,
    )


__all__ = [
    "StorageConfig",
    "ModuleSpec",
    "EdgeSpec",
    "UnimemConfig",
    "load_config",
    "load_unimem",
    "build_storage",
    "build_graph",
]
