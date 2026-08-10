"""Declarative graph construction.

``GraphSpec`` / ``NodeSpec`` / ``EdgeSpec`` are dataclass specs that describe
*what* to build; :class:`MemoryGraphBuilder` resolves them against a
:class:`~unimem.factory.registry.Registry` to instantiate modules + policies
and assemble a :class:`~unimem.graph.graph.MemoryGraph`.

The specs are intentionally JSON-ish: every field is a primitive, so a spec
can round-trip through ``json.loads`` (or YAML) for experiment-driven
architecture search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..core.module import MemoryModule
from ..core.slots import MemorySlot
from ..factory.registry import Registry
from .edge import EdgeKind, MemoryEdge
from .graph import MemoryGraph
from .node import MemoryNode


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
@dataclass
class NodeSpec:
    """Declarative description of one graph node."""

    node_id: str
    slot: Union[MemorySlot, str]
    impl: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.slot, MemorySlot):
            self.slot = MemorySlot.from_value(self.slot)


@dataclass
class EdgeSpec:
    """Declarative description of one graph edge.

    Policies are referenced by ``(policy_type, name)`` so the registry stays
    the single source of truth. ``policy_kwargs`` lets a spec customise the
    policy at construction time without registering a new class.
    """

    source: str
    target: str
    kind: Union[EdgeKind, str]
    policy_type: Optional[str] = None  # "write" | "read" | "consolidation" | "forget"
    policy_name: Optional[str] = None
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EdgeKind):
            self.kind = EdgeKind.from_value(self.kind)


@dataclass
class GraphSpec:
    """Top-level declarative description of a memory graph."""

    nodes: List[NodeSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)
    default_write_policy: Optional[Dict[str, Any]] = None
    default_read_policy: Optional[Dict[str, Any]] = None
    default_forget_policy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphSpec":
        """Build a :class:`GraphSpec` from a dict (e.g. parsed from JSON).

        Dict shape::

            {
              "nodes": [{"node_id": ..., "slot": ..., "impl": ..., "kwargs": {...}}, ...],
              "edges": [{"source": ..., "target": ..., "kind": "feeds",
                         "policy_type": "write", "policy_name": "always"}, ...],
              "default_write_policy": {"name": "always"},
              "default_read_policy":  {"name": "concat"},
              "default_forget_policy": {"name": "noop"},
              "metadata": {...}
            }
        """
        nodes = [NodeSpec(**n) for n in d.get("nodes", [])]
        edges = [EdgeSpec(**e) for e in d.get("edges", [])]
        return cls(
            nodes=nodes,
            edges=edges,
            default_write_policy=d.get("default_write_policy"),
            default_read_policy=d.get("default_read_policy"),
            default_forget_policy=d.get("default_forget_policy"),
            metadata=dict(d.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip back to a plain dict."""
        from dataclasses import asdict

        def _normalize(value: Any) -> Any:
            if isinstance(value, MemorySlot):
                return value.value
            if isinstance(value, EdgeKind):
                return value.value
            return value

        d = asdict(self)
        # Normalise enum-valued fields to strings.
        for n in d["nodes"]:
            n["slot"] = _normalize(n["slot"])
        for e in d["edges"]:
            e["kind"] = _normalize(e["kind"])
        return d


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class MemoryGraphBuilder:
    """Resolve a :class:`GraphSpec` against a registry into a ``MemoryGraph``."""

    def __init__(self, registry: Optional[Registry] = None) -> None:
        self.registry = registry or Registry()

    # -- public API ------------------------------------------------------ #
    def build(self, spec: Union[GraphSpec, Dict[str, Any]]) -> MemoryGraph:
        """Construct a :class:`MemoryGraph` from a spec or dict."""
        if isinstance(spec, dict):
            spec = GraphSpec.from_dict(spec)
        if not isinstance(spec, GraphSpec):
            raise TypeError(
                f"build expected GraphSpec or dict, got {type(spec).__name__}"
            )

        graph = MemoryGraph(
            default_write_policy=self._maybe_policy(spec.default_write_policy, "write"),
            default_read_policy=self._maybe_policy(spec.default_read_policy, "read"),
            default_forget_policy=self._maybe_policy(spec.default_forget_policy, "forget"),
        )

        # Nodes first so edges can reference them.
        for node_spec in spec.nodes:
            graph.add_node(self._build_node(node_spec))

        for edge_spec in spec.edges:
            graph.add_edge(self._build_edge(edge_spec))

        return graph

    # -- internals ------------------------------------------------------- #
    def _build_node(self, spec: NodeSpec) -> MemoryNode:
        module = self.registry.create_module(spec.slot, spec.impl, **spec.kwargs)
        return MemoryNode(
            node_id=spec.node_id,
            slot=spec.slot,
            module=module,
            label=spec.label,
            metadata=dict(spec.metadata),
        )

    def _build_edge(self, spec: EdgeSpec) -> MemoryEdge:
        policy: Optional[Any] = None
        if spec.policy_type and spec.policy_name:
            policy = self.registry.create_policy(
                spec.policy_type, spec.policy_name, **spec.policy_kwargs
            )
        return MemoryEdge(
            source_id=spec.source,
            target_id=spec.target,
            kind=spec.kind,
            policy=policy,
            metadata=dict(spec.metadata),
        )

    def _maybe_policy(
        self, spec: Optional[Dict[str, Any]], policy_type: str
    ) -> Optional[Any]:
        """Resolve a default-policy dict like ``{"name": "always", "kwargs": {}}``."""
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise TypeError(
                f"Default policy spec must be a dict, got {type(spec).__name__}"
            )
        # Two accepted shapes: {"name": "x", "kwargs": {...}} or {"impl": "x"}.
        name = spec.get("name") or spec.get("impl")
        if name is None:
            raise ValueError(
                f"Default {policy_type} policy spec missing 'name'/'impl': {spec!r}"
            )
        kwargs = spec.get("kwargs", {})
        return self.registry.create_policy(policy_type, name, **kwargs)


__all__ = ["GraphSpec", "NodeSpec", "EdgeSpec", "MemoryGraphBuilder"]
