"""``MemoryGraph`` — the orchestrator implementing the three core algorithms.

* :meth:`read`                  — fan-out read
* :meth:`write`                 — fan-in write (BFS along FEEDS edges)
* :meth:`update_all`            — per-step maintenance on every node
* :meth:`run_consolidation_pass` — sedimentation along CONSOLIDATES_TO edges
                                   + forget sweep

Graphs are intentionally small (3–8 nodes per design), so we use plain
adjacency dicts and linear scans. No networkx, no caching, no async.

When a :class:`~unimem.graph_storage.base.GraphStorage` is attached, every
topology mutation (``add_node`` / ``add_edge``) and every write is
double-written to the storage backend. An optional
:class:`~unimem.op_log.OpLog` records each write as a WAL entry for
crash-recovery / audit.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry
from ..core.module import MemoryModule
from ..core.query import Query, QueryResult
from ..core.slots import MemorySlot
from ..policies.consolidation_policy import ConsolidationPolicy, Passthrough
from ..policies.forget_policy import ForgetPolicy, NoOp
from ..policies.read_policy import ConcatRead, ReadPolicy
from ..policies.write_policy import AlwaysWrite, WritePolicy
from .edge import EdgeKind, MemoryEdge
from .node import MemoryNode


class MemoryGraph:
    """An editable directed graph of memory modules.

    Construction is via :meth:`add_node` / :meth:`add_edge`; for declarative
    builds see :class:`~unimem.graph.builder.MemoryGraphBuilder`.
    """

    def __init__(
        self,
        *,
        graph_storage: Optional[Any] = None,
        op_log: Optional[Any] = None,
        default_write_policy: Optional[WritePolicy] = None,
        default_read_policy: Optional[ReadPolicy] = None,
        default_forget_policy: Optional[ForgetPolicy] = None,
    ) -> None:
        self._nodes: Dict[str, MemoryNode] = {}
        # Adjacency: source_id -> list of outgoing edges.
        self._out: Dict[str, List[MemoryEdge]] = {}
        # Reverse adjacency: target_id -> list of incoming edges.
        self._in: Dict[str, List[MemoryEdge]] = {}

        self.default_write_policy: WritePolicy = default_write_policy or AlwaysWrite()
        self.default_read_policy: ReadPolicy = default_read_policy or ConcatRead()
        self.default_forget_policy: ForgetPolicy = default_forget_policy or NoOp()

        # Persistence — optional. When present, every topology mutation and
        # every write/read is mirrored into the storage backend.
        self._graph_storage = graph_storage
        self._op_log = op_log

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def add_node(self, node: MemoryNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])
        # Mirror into storage
        if self._graph_storage is not None:
            self._graph_storage.add_module_node(
                node.node_id,
                node.slot,
                type(node.module).__name__,
                label=node.label,
                **dict(node.metadata),
            )

    def add_edge(self, edge: MemoryEdge) -> None:
        for nid in (edge.source_id, edge.target_id):
            if nid not in self._nodes:
                raise KeyError(f"Edge references unknown node: {nid!r}")
        self._out[edge.source_id].append(edge)
        self._in[edge.target_id].append(edge)
        # Mirror into storage
        if self._graph_storage is not None:
            policy_dict = _policy_to_dict(edge.policy)
            self._graph_storage.add_module_edge(
                edge.source_id, edge.target_id, edge.kind, policy_dict
            )

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self._nodes.get(node_id)

    def nodes(self) -> List[MemoryNode]:
        return list(self._nodes.values())

    def edges(self) -> List[MemoryEdge]:
        out: List[MemoryEdge] = []
        for es in self._out.values():
            out.extend(es)
        return out

    def edges_of(self, node_id: str, kind: Optional[EdgeKind] = None) -> List[MemoryEdge]:
        """Outgoing edges of ``node_id`` optionally filtered by kind."""
        return [e for e in self._out.get(node_id, []) if kind is None or e.kind == kind]

    def edges_into(self, node_id: str, kind: Optional[EdgeKind] = None) -> List[MemoryEdge]:
        """Incoming edges of ``node_id`` optionally filtered by kind."""
        return [e for e in self._in.get(node_id, []) if kind is None or e.kind == kind]

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------ #
    # Tree queries (SUBSUMES edges form a forest / DAG)
    # ------------------------------------------------------------------ #
    def get_children(self, node_id: str) -> List[str]:
        """Ids of nodes this node SUBSUMES (direct children only)."""
        return [
            e.target_id
            for e in self._out.get(node_id, [])
            if e.kind == EdgeKind.SUBSUMES
        ]

    def get_parent(self, node_id: str) -> Optional[str]:
        """The node that SUBSUMES this one, if any. First match wins."""
        for e in self._in.get(node_id, []):
            if e.kind == EdgeKind.SUBSUMES:
                return e.source_id
        return None

    def get_subtree(self, node_id: str) -> List[str]:
        """All descendants via SUBSUMES edges, BFS order, excluding ``node_id``."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        seen: Set[str] = set()
        order: List[str] = []
        dq = deque(self.get_children(node_id))
        while dq:
            cur = dq.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            order.append(cur)
            dq.extend(self.get_children(cur))
        return order

    # ------------------------------------------------------------------ #
    # Algorithm 1: fan-out read
    # ------------------------------------------------------------------ #
    def read(self, query: Query) -> List[QueryResult]:
        """Broadcast ``query`` to every node it targets.

        Target selection:

        * If ``query.slot_filter`` is set, only nodes whose ``slot`` is in
          that set are queried.
        * Otherwise every node is queried.

        Each node returns one :class:`QueryResult` stamped with
        ``source_node_id`` and ``source_slot``. The list preserves the
        graph's insertion order of nodes.

        To collapse results into a single one, call
        :meth:`merge_results` or use the graph-level ``default_read_policy``.
        """
        results: List[QueryResult] = []
        for node in self._nodes.values():
            if not query.accepts_slot(node.slot):
                continue
            r = node.module.read(query)
            if r is None:
                r = QueryResult()
            # Stamp provenance (do not overwrite if module already filled it in).
            r.source_node_id = r.source_node_id or node.node_id
            r.source_slot = r.source_slot or node.slot.value
            if query.top_k is not None:
                r.truncate(query.top_k)
            results.append(r)
        return results

    def merge_results(self, results: List[QueryResult]) -> QueryResult:
        """Combine fan-out read results via the graph-level read policy."""
        return self.default_read_policy.merge(results)

    # ------------------------------------------------------------------ #
    # Algorithm 2: fan-in write (BFS along FEEDS edges)
    # ------------------------------------------------------------------ #
    def write(
        self,
        entry: MemoryEntry,
        context: MemoryContext,
        source_node_id: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Write ``entry`` and propagate it along FEEDS edges.

        Behaviour:

        * If ``source_node_id`` is provided: write there first; if the
          (effective) write policy rejects the entry, propagation stops.
        * Otherwise: write to every "root" node — i.e. nodes with no
          incoming FEEDS edge. If there are none, write to every node.
        * BFS then walks outgoing FEEDS edges. Per-edge ``WritePolicy``
          (in ``edge.policy``) gates propagation; if absent, the source
          node's policy applies; if that is also absent, the graph's
          ``default_write_policy`` applies.
        * Each target node is visited at most once (VISITED set guards
          against cycles). The same in-memory ``entry`` object is passed
          unchanged (identity propagation); any transformation is the
          target module's own ``write`` responsibility.

        When an :class:`~unimem.op_log.OpLog` is attached, every write is
        recorded as a WAL entry before dispatch (and a post-write marker
        after), enabling crash recovery and audit.

        Returns a ``{node_id: did_store}`` map for every node that was
        *reached* (policies may still have rejected the write at the gate).
        """
        if not self._nodes:
            return {}

        # Stamp provenance on the entry if not set.
        if entry.source_slot is None:
            entry.source_slot = (
                self._nodes[source_node_id].slot.value
                if source_node_id and source_node_id in self._nodes
                else None
            )

        # WAL pre-write
        if self._op_log is not None:
            from ..op_log import OpLogEntry
            self._op_log.append(OpLogEntry(
                op_type="write",
                node_id=source_node_id,
                entry_dict=_entry_to_dict(entry),
                context_dict=_context_to_dict(context),
            ))

        results: Dict[str, bool] = {}
        visited: Set[str] = set()
        queue: deque = deque()

        # Seed the queue.
        if source_node_id is not None:
            if source_node_id not in self._nodes:
                raise KeyError(f"Unknown source_node_id: {source_node_id!r}")
            queue.append(source_node_id)
        else:
            roots = self._root_feeds_nodes()
            seeds = roots if roots else list(self._nodes.keys())
            queue.extend(seeds)

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self._nodes[node_id]
            # The write policy applied at *this* node is determined by who
            # pointed at it. For seed nodes, no FEEDS edge exists, so use
            # the node's own policy or the graph default.
            edge_into = self._first_feeds_edge_into(node_id)
            policy = self._effective_write_policy(node.module, edge_into)
            if not policy.should_write(node.module, entry, context):
                results[node_id] = False
                continue

            stored = bool(node.module.write(entry, context))
            results[node_id] = stored

            # Enqueue FEEDS successors regardless of whether we stored; if
            # a downstream edge policy is more permissive it can still
            # propagate. (Re-evaluation happens at dequeue time.)
            for edge in self._out.get(node_id, []):
                if edge.kind == EdgeKind.FEEDS and edge.target_id not in visited:
                    queue.append(edge.target_id)

        # WAL post-write marker
        if self._op_log is not None:
            from ..op_log import OpLogEntry
            self._op_log.append(OpLogEntry(
                op_type="write_done",
                node_id=source_node_id,
                result=results,
            ))

        return results

    def _root_feeds_nodes(self) -> List[str]:
        """Nodes with no incoming FEEDS edge."""
        return [
            nid
            for nid in self._nodes
            if not any(e.kind == EdgeKind.FEEDS for e in self._in.get(nid, []))
        ]

    def _first_feeds_edge_into(self, node_id: str) -> Optional[MemoryEdge]:
        for e in self._in.get(node_id, []):
            if e.kind == EdgeKind.FEEDS:
                return e
        return None

    def _effective_write_policy(
        self, module: MemoryModule, edge: Optional[MemoryEdge]
    ) -> WritePolicy:
        """Edge policy > module policy > graph default."""
        if edge is not None and isinstance(edge.policy, WritePolicy):
            return edge.policy
        mod_pol = getattr(module, "write_policy", None)
        if isinstance(mod_pol, WritePolicy):
            return mod_pol
        return self.default_write_policy

    # ------------------------------------------------------------------ #
    # Algorithm 3: consolidation pass
    # ------------------------------------------------------------------ #
    def run_consolidation_pass(
        self, context: MemoryContext
    ) -> Dict[str, int]:
        """Sediment along CONSOLIDATES_TO edges, then forget.

        For every CONSOLIDATES_TO edge (source -> target):

        * If the edge carries a ``ConsolidationPolicy``, use it; otherwise
          wrap the source module's own ``consolidate`` via :class:`Passthrough`.
        * Extract entries and write them into the target module.
        * Count both extracted and stored counts.

        After all edges are processed, the graph-level forget policy is
        applied to every node (per-module forget policy overrides if set).

        Returns a dict with keys ``extracted``, ``stored``, ``forgotten``
        each mapping ``node_id -> count`` for inspection / debugging.
        """
        extracted: Dict[str, int] = {nid: 0 for nid in self._nodes}
        stored: Dict[str, int] = {nid: 0 for nid in self._nodes}
        forgotten: Dict[str, int] = {nid: 0 for nid in self._nodes}

        for edge in self.edges():
            if edge.kind != EdgeKind.CONSOLIDATES_TO:
                continue
            source = self._nodes.get(edge.source_id)
            target = self._nodes.get(edge.target_id)
            if source is None or target is None:
                continue

            policy: ConsolidationPolicy
            if isinstance(edge.policy, ConsolidationPolicy):
                policy = edge.policy
            else:
                policy = Passthrough()

            new_entries = policy.extract(source.module, target.module, context)
            extracted[source.node_id] = extracted.get(source.node_id, 0) + len(new_entries)

            for e in new_entries:
                if target.module.write(e, context):
                    stored[target.node_id] = stored.get(target.node_id, 0) + 1

        # Forget sweep
        for node in self._nodes.values():
            pol = getattr(node.module, "forget_policy", None)
            if not isinstance(pol, ForgetPolicy):
                pol = self.default_forget_policy
            n = pol.apply(node.module, context)
            forgotten[node.node_id] = forgotten.get(node.node_id, 0) + n

        return {"extracted": extracted, "stored": stored, "forgotten": forgotten}

    # ------------------------------------------------------------------ #
    # Per-step maintenance
    # ------------------------------------------------------------------ #
    def update_all(self, context: MemoryContext) -> None:
        """Invoke ``module.update(context)`` on every node."""
        for node in self._nodes.values():
            node.module.update(context)

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """High-level diagnostic info: node count, edge counts, per-node stats."""
        per_kind: Dict[str, int] = {k.name: 0 for k in EdgeKind}
        for e in self.edges():
            per_kind[e.kind.name] += 1
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self.edges()),
            "edges_by_kind": per_kind,
            "nodes": {
                nid: {
                    "slot": n.slot.name,
                    "label": n.label,
                    "stats": _safe_stats(n.module),
                }
                for nid, n in self._nodes.items()
            },
        }

    # ------------------------------------------------------------------ #
    # Storage accessors
    # ------------------------------------------------------------------ #
    @property
    def graph_storage(self):
        return self._graph_storage

    @property
    def op_log(self):
        return self._op_log


def _safe_stats(module: MemoryModule) -> Dict[str, Any]:
    try:
        return module.stats()
    except Exception as exc:  # pragma: no cover - defensive
        return {"_error": repr(exc)}


def _policy_to_dict(policy: Any) -> Optional[Dict[str, Any]]:
    """Serialise an edge policy to a dict for storage."""
    if policy is None:
        return None
    # WritePolicy / ConsolidationPolicy: best-effort serialise
    if hasattr(policy, "to_dict") and callable(policy.to_dict):
        try:
            return policy.to_dict()
        except Exception:
            pass
    return {"type": type(policy).__name__}


def _entry_to_dict(entry: MemoryEntry) -> Dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "text": entry.text,
        "semantic_keys": list(entry.semantic_keys),
        "spatial_keys": [list(k) for k in entry.spatial_keys],
        "temporal_keys": list(entry.temporal_keys),
        "metadata": dict(entry.metadata),
        "source_slot": entry.source_slot,
    }


def _context_to_dict(ctx: MemoryContext) -> Dict[str, Any]:
    return {
        "episode_id": ctx.episode_id,
        "task_id": ctx.task_id,
        "pose": list(ctx.pose) if ctx.pose is not None else None,
        "timestamp": ctx.timestamp,
        "step": ctx.step,
        "extra": dict(ctx.extra),
    }


__all__ = ["MemoryGraph"]
