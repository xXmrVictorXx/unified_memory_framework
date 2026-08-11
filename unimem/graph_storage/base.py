"""``GraphStorage`` — graph database abstraction for unimem.

Two-layer API:

* **Low-level CRUD** (``add_node`` / ``add_edge`` / ``get_node`` / ...) —
  backend-agnostic operations on a property graph. Concrete backends
  implement these in terms of their native primitives.
* **Memory-level convenience** (``add_memory_node`` / ``add_time_index`` /
  ``query_memories`` / ``add_module_node`` / ...) — sensible default
  implementations that build on the low-level API and encode unimem's
  node-label conventions (slot labels, TimeIndex nodes, AT_TIME edges).

Node label conventions
----------------------

* ``MemorySlot.value`` (e.g. ``"working_memory"``, ``"scene_graph"``) is the
  primary label on a memory node.
* ``"TimeIndex"`` labels time-anchor nodes connected to memory nodes via
  ``AT_TIME`` relationships.
* ``"ModuleNode"`` labels the outer unimem graph topology nodes (one per
  :class:`~unimem.graph.node.MemoryNode`).
* Reproduction methods are free to add their own sub-labels (``"Person"``,
  ``"Object"``, ``"Activity"``, ``"r4_object"`` ...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.entry import MemoryEntry
from ..core.slots import MemorySlot
from ..graph.edge import EdgeKind


# --------------------------------------------------------------------------- #
# Property (de)serialisation helpers — shared across every backend
# --------------------------------------------------------------------------- #
def _entry_to_props(entry: MemoryEntry) -> Dict[str, Any]:
    """Flatten a :class:`MemoryEntry` into node properties.

    Lists are preserved as-is for in-memory backends; backends that cross a
    serialisation boundary (e.g. Neo4j) are responsible for converting lists
    to a storable form.
    """
    return {
        "entry_id": entry.entry_id,
        "text": entry.text,
        "semantic_keys": list(entry.semantic_keys),
        "spatial_keys": [list(k) for k in entry.spatial_keys],
        "temporal_keys": list(entry.temporal_keys),
        "metadata": dict(entry.metadata),
        "source_slot": entry.source_slot,
    }


def _props_to_entry(props: Dict[str, Any]) -> MemoryEntry:
    """Reconstruct a :class:`MemoryEntry` from node properties."""
    return MemoryEntry(
        entry_id=props.get("entry_id") or "",
        text=props.get("text") or "",
        semantic_keys=props.get("semanticKeys") or props.get("semantic_keys") or [],
        spatial_keys=[
            tuple(float(c) for c in k)
            for k in (props.get("spatialKeys") or props.get("spatial_keys") or [])
        ],
        temporal_keys=[
            float(t)
            for t in (props.get("temporalKeys") or props.get("temporal_keys") or [])
        ],
        metadata=dict(props.get("metadata") or {}),
        source_slot=props.get("source_slot"),
    )


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #
class GraphStorage(ABC):
    """Abstract graph storage.

    Concrete implementations must override every ``@abstractmethod`` below.
    The memory-level convenience methods have working default implementations
    written in terms of the low-level API; override them only when a backend
    has a more efficient native path (e.g. a Cypher MATCH for
    ``query_memories``).
    """

    # ------------------------------------------------------------------ #
    # Low-level CRUD — concrete backends MUST implement
    # ------------------------------------------------------------------ #
    @abstractmethod
    def add_node(
        self, node_id: str, labels: List[str], properties: Dict[str, Any]
    ) -> bool:
        """Insert (or MERGE) a node. Returns True on success.

        If ``node_id`` already exists, the implementation should update
        properties and merge labels (idempotent upsert).
        """

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert (or MERGE) a typed relationship. Returns True on success."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{"labels": [...], "properties": {...}}`` or None."""

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return ``[(other_id, rel_type, properties), ...]``.

        ``direction`` ∈ ``{"out", "in", "both"}``.
        """

    @abstractmethod
    def delete_node(self, node_id: str) -> int:
        """Delete a node and (optionally) its attached edges. Returns count."""

    @abstractmethod
    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: Optional[str] = None,
    ) -> int:
        """Delete matching edges. Returns count deleted."""

    @abstractmethod
    def query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a backend-native query (e.g. Cypher).

        Backends that do not support a query language should emulate the
        common subset (MATCH by label/property, WHERE on properties, RETURN
        nodes or relationships). The in-memory backend implements a small
        Cypher-like DSL sufficient for unimem's needs.
        """

    @abstractmethod
    def bfs(
        self,
        start_id: str,
        rel_types: Optional[Sequence[str]] = None,
        max_depth: int = 100,
    ) -> List[str]:
        """BFS from ``start_id`` over edges whose type is in ``rel_types``.

        Returns reachable node ids (excluding ``start_id``).
        """

    # ------------------------------------------------------------------ #
    # Memory-level convenience — defaults provided
    # ------------------------------------------------------------------ #
    def add_memory_node(
        self,
        slot: MemorySlot,
        entry: MemoryEntry,
        extra_labels: Optional[List[str]] = None,
    ) -> bool:
        """Store ``entry`` as a node labelled with ``slot.value``."""
        labels = [slot.value] + (extra_labels or [])
        return self.add_node(entry.entry_id, labels, _entry_to_props(entry))

    def add_time_index(self, memory_node_id: str, **time_props: Any) -> bool:
        """Attach a :TimeIndex node to ``memory_node_id`` via ``:AT_TIME``.

        ``time_props`` becomes the TimeIndex node's properties. Suggested
        keys: ``timestamp``, ``clip_index``, ``period``, ``start_t``, ``end_t``.
        """
        if not time_props:
            raise ValueError("add_time_index requires at least one time property")
        time_id = self._time_index_id(memory_node_id, time_props)
        self.add_node(time_id, ["TimeIndex"], dict(time_props))
        return self.add_edge(memory_node_id, time_id, "AT_TIME", {})

    def query_memories(
        self,
        slot: Optional[MemorySlot] = None,
        semantic_keys: Optional[Sequence[str]] = None,
        time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        clip_index: Optional[int] = None,
        extra_label: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Convenience filter: slot label + semantic keys + time window.

        Default implementation walks every node by label and applies filters
        in Python. Backends with a native query language should override.
        """
        # Collect candidates by label.
        if slot is not None:
            label = slot.value
        elif extra_label is not None:
            label = extra_label
        else:
            label = None

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        # Use the backend's native query if available; else fall back to
        # scanning nodes via get_node on a known id set is impossible, so we
        # rely on a "list all node ids" query.
        for node_id, labels, props in self._iter_nodes():
            if label is not None and label not in labels:
                continue
            # When filtering by both slot and extra_label, ensure both present
            if (
                slot is not None
                and extra_label is not None
                and extra_label not in labels
            ):
                continue
            # Skip TimeIndex nodes themselves.
            if "TimeIndex" in labels:
                continue
            # Skip ModuleNode topology nodes (they also carry the slot label
            # via add_module_node, but they aren't memory entries).
            if "ModuleNode" in labels:
                continue
            candidates.append((node_id, props))

        # Apply semantic_keys filter (any-match semantics — at least one key).
        if semantic_keys:
            wanted = set(semantic_keys)
            candidates = [
                (nid, p)
                for (nid, p) in candidates
                if wanted & set(p.get("semantic_keys") or p.get("semanticKeys") or [])
            ]

        # Time-based filters.
        if time_range is not None or clip_index is not None:
            candidates = self._filter_by_time(
                candidates, time_range, clip_index
            )

        entries = []
        for (nid, p) in candidates:
            props = dict(p)
            props.setdefault("entry_id", nid)
            entries.append(_props_to_entry(props))
        if top_k is not None:
            entries = entries[:top_k]
        return entries

    def get_time_indexed_nodes(
        self,
        time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        clip_index: Optional[int] = None,
        period: Optional[str] = None,
        slot: Optional[MemorySlot] = None,
    ) -> List[MemoryEntry]:
        """Return memory nodes attached to a TimeIndex matching the filters.

        Resolves ``TimeIndex`` → ``:AT_TIME`` ← memory node.
        """
        # Iterate TimeIndex nodes.
        out: List[MemoryEntry] = []
        for node_id, labels, props in self._iter_nodes():
            if "TimeIndex" not in labels:
                continue
            if clip_index is not None and props.get("clip_index") != clip_index:
                continue
            if period is not None and props.get("period") != period:
                continue
            if time_range is not None:
                t = props.get("timestamp")
                if t is None:
                    # Try start_t / end_t window overlap
                    s = props.get("start_t")
                    e = props.get("end_t")
                    if s is not None and e is not None:
                        lo, hi = time_range
                        if lo is not None and float(e) < float(lo):
                            continue
                        if hi is not None and float(s) > float(hi):
                            continue
                    else:
                        continue
                else:
                    lo, hi = time_range
                    if lo is not None and float(t) < float(lo):
                        continue
                    if hi is not None and float(t) > float(hi):
                        continue
            # Find the memory node pointing at this TimeIndex.
            for src_id, rel_type, _ in self.get_neighbors(node_id, "AT_TIME", "in"):
                node = self.get_node(src_id)
                if node is None:
                    continue
                node_labels = node.get("labels", [])
                if slot is not None and slot.value not in node_labels:
                    continue
                if "TimeIndex" in node_labels:
                    continue
                # Reconstruct entry, ensuring entry_id defaults to node id.
                entry_props = dict(node["properties"])
                entry_props.setdefault("entry_id", src_id)
                out.append(_props_to_entry(entry_props))
        return out

    # ------------------------------------------------------------------ #
    # Outer-graph topology (used by MemoryGraph persistence)
    # ------------------------------------------------------------------ #
    def add_module_node(
        self,
        node_id: str,
        slot: MemorySlot,
        impl: str,
        **props: Any,
    ) -> bool:
        """Persist a :class:`MemoryNode` as ``:ModuleNode:<slot>``."""
        labels = ["ModuleNode", slot.value]
        properties = {"node_id": node_id, "slot": slot.value, "impl": impl, **props}
        return self.add_node(node_id, labels, properties)

    def add_module_edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        policy: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persist a :class:`MemoryEdge` as ``<EdgeKind.value>`` relationship."""
        kind_enum = EdgeKind.from_value(kind) if not isinstance(kind, EdgeKind) else kind
        return self.add_edge(source, target, kind_enum.value, policy or {})

    # ------------------------------------------------------------------ #
    # Hooks for backends — override these for efficient enumeration
    # ------------------------------------------------------------------ #
    def _iter_nodes(self) -> List[Tuple[str, List[str], Dict[str, Any]]]:
        """Return ``[(node_id, labels, properties), ...]`` for every node.

        Default implementation issues a backend-agnostic MATCH via
        :meth:`query`. Backends with a more efficient scan should override.
        """
        rows = self.query("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, n")
        out: List[Tuple[str, List[str], Dict[str, Any]]] = []
        for row in rows:
            node_id = row.get("id") or row.get("node_id")
            if node_id is None:
                continue
            labels = row.get("labels") or []
            props = row.get("properties") or row.get("n") or {}
            out.append((str(node_id), list(labels), dict(props)))
        return out

    def _filter_by_time(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
        time_range: Optional[Tuple[Optional[float], Optional[float]]],
        clip_index: Optional[int],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Default time filter: looks at each node's TimeIndex neighbours."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        for nid, props in candidates:
            keep = True
            # Check attached TimeIndex nodes (fetch the node properties — the
            # edge itself is usually empty).
            time_neighbors = self.get_neighbors(nid, "AT_TIME", "out")
            if not time_neighbors:
                # Fall back to node-level temporal_keys.
                ts = props.get("temporal_keys") or props.get("temporalKeys") or []
                if time_range is not None:
                    lo, hi = time_range
                    if ts:
                        keep = any(
                            (lo is None or float(t) >= lo)
                            and (hi is None or float(t) <= hi)
                            for t in ts
                        )
                    else:
                        keep = False
                if clip_index is not None:
                    keep = keep and (props.get("clip_index") == clip_index)
            else:
                keep = False
                for other_id, _, _ in time_neighbors:
                    ti_node = self.get_node(other_id)
                    if ti_node is None:
                        continue
                    time_props = ti_node["properties"]
                    if clip_index is not None:
                        if time_props.get("clip_index") != clip_index:
                            continue
                    if time_range is not None:
                        lo, hi = time_range
                        t = time_props.get("timestamp")
                        if t is not None:
                            if lo is not None and float(t) < lo:
                                continue
                            if hi is not None and float(t) > hi:
                                continue
                        else:
                            s = time_props.get("start_t")
                            e = time_props.get("end_t")
                            if s is not None and e is not None:
                                if lo is not None and float(e) < lo:
                                    continue
                                if hi is not None and float(s) > hi:
                                    continue
                            elif lo is not None or hi is not None:
                                continue
                    keep = True
                    break
            if keep:
                out.append((nid, props))
        return out

    @staticmethod
    def _time_index_id(memory_node_id: str, time_props: Dict[str, Any]) -> str:
        """Deterministic id for a TimeIndex node."""
        items = sorted((str(k), str(v)) for k, v in time_props.items())
        return f"time-{memory_node_id}-{hash(tuple(items)) & 0xFFFFFFFFFF:x}"


__all__ = ["GraphStorage"]
