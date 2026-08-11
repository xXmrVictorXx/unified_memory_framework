"""CLiViS ``RelationGraph`` — storage-backed property-graph facade.

A thin facade over :class:`~unimem.graph_storage.base.GraphStorage` that
preserves CLiViS's high-level API (``add_person`` / ``add_relation`` /
``add_action`` / ``extract_subgraph_by_nodes`` / ...) while delegating
storage to a portable backend.

Differences from the legacy implementation:

* **No slot ABC inheritance** — the class subclasses
  :class:`~unimem.core.module.MemoryModule` (so it can sit in a unimem
  :class:`~unimem.graph.graph.MemoryGraph`) but does **not** inherit any
  slot ABC. Pipeline-specific behaviour is encoded in the high-level
  methods (``add_person``, ``add_action``, ...), not in slot-ABC contracts.
* **Nodes live in :class:`GraphStorage`** with labels ``scene_graph``,
  the CLiViS-specific sub-label (``Person`` / ``Object`` / ``Area`` /
  ``Activity``), and arbitrary properties.
* **Edges are typed relationships** in the same GraphStorage (``PERFORMS``,
  ``AFFECTS``, ``USES``, ``FROM``, ``TO``, ``NEXT_ACTION``, ``CONTAINS``,
  plus any freeform type the LLM extracts).
* **Time-indexed entities** (Areas, Activities) carry ``time_range`` props;
  callers wanting graph-native time queries can additionally attach a
  ``:TimeIndex`` node via the helpers in :mod:`unimem.graph_storage.time_index`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage


# --------------------------------------------------------------------------- #
# Node labels (mirror the original Neo4j labels)
# --------------------------------------------------------------------------- #
class NodeLabels:
    """String constants for the CLiViS entity labels."""

    PERSON = "Person"
    OBJECT = "Object"
    ACTIVITY = "Activity"
    AREA = "Area"

    _ALL = (PERSON, OBJECT, ACTIVITY, AREA)

    @classmethod
    def from_value(cls, value: str) -> str:
        """Validate ``value`` is a known CLiViS label (case-insensitive)."""
        norm = str(value).strip().lower()
        for label in cls._ALL:
            if label.lower() == norm:
                return label
        raise KeyError(f"Unknown NodeLabels value: {value!r}")


# Standard typed relationship names used by ``add_action``
ACTION_REL_TYPES = {
    "agent": "PERFORMS",
    "patient": "AFFECTS",
    "instrument": "USES",
    "source": "FROM",
    "target": "TO",
}


class RelationGraph(MemoryModule):
    """Storage-backed property graph mirroring CLiViS's Neo4j schema."""

    SLOT = MemorySlot.SG  # nodes carry this slot label

    def __init__(self, graph_storage: Optional[GraphStorage] = None) -> None:
        super().__init__(slot=self.SLOT)
        self._gs = graph_storage or InMemoryGraphStorage()

    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #
    def add_update_node(
        self, node_name: str, node_label: str, attr_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not node_name:
            return False
        try:
            label = NodeLabels.from_value(node_label)
        except KeyError:
            return False
        labels = [self.SLOT.value, label]
        props = {"name": node_name, "label": label, **(attr_dict or {})}
        # Use MERGE semantics: add_node will upsert.
        self._gs.add_node(node_name, labels, props)
        return True

    def add_person(self, person_name: str, info: str) -> bool:
        return self.add_update_node(person_name, "Person", {"info": info})

    def add_update_objects(
        self, obj_name: str, time_range: str, obj_info: str
    ) -> bool:
        return self.add_update_node(obj_name, "Object", {
            "time_range": time_range, "info": obj_info,
        })

    def add_area(self, area_name: str, time_range: str, area_info: str) -> bool:
        return self.add_update_node(area_name, "Area", {
            "time_range": time_range, "info": area_info,
        })

    def get_node_info(
        self, node_name: str, node_label: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        node = self._gs.get_node(node_name)
        if node is None:
            return None
        labels = node["labels"]
        if self.SLOT.value not in labels:
            return None
        props = dict(node["properties"])
        # Pick the CLiViS-specific label (anything not the slot label)
        clivis_labels = [l for l in labels if l != self.SLOT.value]
        primary = clivis_labels[0] if clivis_labels else props.get("label")
        if node_label is not None:
            wanted = NodeLabels.from_value(node_label)
            if wanted != primary:
                return None
        return {"name": node_name, "label": primary, "props": props}

    # ------------------------------------------------------------------ #
    # Relationship operations
    # ------------------------------------------------------------------ #
    def add_relation(
        self,
        node_a_name: str,
        node_b_name: str,
        relation_type: str,
        relation_info: str = "",
        start_time: str = "",
        end_time: Optional[str] = None,
    ) -> bool:
        if self._gs.get_node(node_a_name) is None or self._gs.get_node(node_b_name) is None:
            return False
        return self._gs.add_edge(
            node_a_name,
            node_b_name,
            relation_type,
            {
                "info": relation_info,
                "start_time": start_time,
                "end_time": end_time if end_time is not None else "",
            },
        )

    def get_relation_info(self, relation_name: str) -> Optional[Dict[str, Any]]:
        # Find first edge with the given type.
        for node_id, _, _ in self._iter_nodes():
            for other_id, rtype, props in self._gs.get_neighbors(node_id, direction="out"):
                if rtype == relation_name:
                    return {
                        "source": node_id,
                        "target": other_id,
                        "type": rtype,
                        "props": dict(props),
                    }
        return None

    def get_relations_of_node(
        self, node_name: str, node_label: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        if self._gs.get_node(node_name) is None:
            return {"outgoing": [], "incoming": []}
        outgoing = [
            {"type": rtype, "endpoint": oid, "props": dict(props)}
            for oid, rtype, props in self._gs.get_neighbors(node_name, direction="out")
        ]
        incoming = [
            {"type": rtype, "endpoint": oid, "props": dict(props)}
            for oid, rtype, props in self._gs.get_neighbors(node_name, direction="in")
        ]
        return {"outgoing": outgoing, "incoming": incoming}

    def get_paths_between_nodes(
        self,
        node_a_name: str,
        node_b_name: str,
        max_step: int = 10,
        dual_direction: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """All simple paths A → B up to ``max_step`` hops."""
        if self._gs.get_node(node_a_name) is None or self._gs.get_node(node_b_name) is None:
            return []
        results: List[List[Dict[str, Any]]] = []

        def neighbours(n: str) -> List[Tuple[str, str]]:
            out = [(oid, rtype) for oid, rtype, _ in self._gs.get_neighbors(n, direction="out")]
            if dual_direction:
                out += [(oid, rtype) for oid, rtype, _ in self._gs.get_neighbors(n, direction="in")]
            return out

        # DFS
        stack: List[Tuple[str, List[str], List[Dict[str, Any]]]] = [
            (node_a_name, [node_a_name], [])
        ]
        while stack:
            current, path, edge_path = stack.pop()
            if len(edge_path) >= max_step:
                continue
            for nb, rtype in neighbours(current):
                # For directional edges, source/target are clear; for
                # reverse-traversed incoming edges, swap so "from" is current.
                rels = self._gs.get_neighbors(current, rtype, "out")
                direction = "out" if any(o == nb for o, _, _ in rels) else "in"
                if direction == "out":
                    step_from, step_to = current, nb
                else:
                    step_from, step_to = nb, current
                if nb in path:
                    continue
                step = list(edge_path) + [{
                    "from": step_from, "to": step_to, "type": rtype,
                }]
                if nb == node_b_name:
                    results.append(step)
                    continue
                stack.append((nb, path + [nb], step))
        return results

    # ------------------------------------------------------------------ #
    # Action operations
    # ------------------------------------------------------------------ #
    def add_action(
        self,
        action_name: str,
        action_info: str,
        time_range: str,
        node_agent_name: Optional[str] = None,
        node_patient_name: Optional[str] = None,
        node_instrument_name: Optional[str] = None,
        node_source_name: Optional[str] = None,
        node_target_name: Optional[str] = None,
        prev_action_id: Optional[str] = None,
    ) -> str:
        action_id = self._compose_action_id(action_name, time_range)
        self.add_update_node(action_id, "Activity", {
            "action_name": action_name,
            "info": action_info,
            "time_range": time_range,
        })
        # Wire role relationships (action → entity)
        for endpoint, rel_type in (
            (node_agent_name, "PERFORMS"),
            (node_patient_name, "AFFECTS"),
            (node_instrument_name, "USES"),
            (node_source_name, "FROM"),
            (node_target_name, "TO"),
        ):
            if endpoint and self._gs.get_node(endpoint) is not None:
                self.add_relation(action_id, endpoint, rel_type, "", time_range, "")
        if prev_action_id is not None and self._gs.get_node(prev_action_id) is not None:
            self.add_relation(prev_action_id, action_id, "NEXT_ACTION", "", time_range, "")
        return action_id

    def get_action_with_relations(self, action_id: str) -> Optional[Dict[str, Any]]:
        info = self.get_node_info(action_id, "Activity")
        if info is None:
            return None
        rels = self.get_relations_of_node(action_id)
        return {
            "action_id": action_id,
            "props": dict(info["props"]),
            "relations": rels,
            "prev_action_id": self._find_prev_action(action_id),
            "next_action_id": self._find_next_action(action_id),
        }

    def _find_prev_action(self, action_id: str) -> Optional[str]:
        for src, rtype, _ in self._gs.get_neighbors(action_id, "NEXT_ACTION", "in"):
            return src
        return None

    def _find_next_action(self, action_id: str) -> Optional[str]:
        for tgt, rtype, _ in self._gs.get_neighbors(action_id, "NEXT_ACTION", "out"):
            return tgt
        return None

    def get_action_chain(self, start_action_id: Optional[str] = None) -> List[str]:
        if start_action_id is not None:
            cursor: Optional[str] = start_action_id
        else:
            # All actions with no prev
            roots: List[str] = []
            for nid, labels, _ in self._iter_nodes():
                if "Activity" not in labels:
                    continue
                if self._find_prev_action(nid) is None:
                    roots.append(nid)
            cursor = roots[0] if roots else None
        chain: List[str] = []
        seen: Set[str] = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            cursor = self._find_next_action(cursor)
        return chain

    def get_actions_related_to_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for src, rtype, _ in self._gs.get_neighbors(entity_name, direction="in"):
            if src == entity_name:
                continue
            src_info = self.get_node_info(src)
            if src_info is None or src_info["label"] != "Activity":
                continue
            entry = self.get_action_with_relations(src)
            if entry is not None:
                out.append(entry)
        return out

    def get_actions_in_period(
        self, time_range: str, agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return actions whose ``time_range`` overlaps the given one."""
        from .navigation_graph import _parse_range_seconds, _ranges_overlap

        target = _parse_range_seconds(time_range)
        out: List[Dict[str, Any]] = []
        for nid, labels, props in self._iter_nodes():
            if "Activity" not in labels:
                continue
            ar = _parse_range_seconds(props.get("time_range", ""))
            if target is None or ar is None or _ranges_overlap(target, ar):
                if agent_name is None:
                    out.append(self.get_action_with_relations(nid))  # type: ignore[arg-type]
                else:
                    # Filter by PERFORMS edge to agent_name
                    perform_targets = [
                        oid for oid, rtype, _ in self._gs.get_neighbors(nid, "PERFORMS", "out")
                    ]
                    if agent_name in perform_targets:
                        out.append(self.get_action_with_relations(nid))  # type: ignore[arg-type]
        return [a for a in out if a is not None]

    def get_all_actions_in_time_range(self, time_range: str) -> List[str]:
        actions = self.get_actions_in_period(time_range)
        chain = self.get_action_chain()
        ordered_ids = [aid for aid in chain if any(a["action_id"] == aid for a in actions)]
        for a in actions:
            if a["action_id"] not in ordered_ids:
                ordered_ids.append(a["action_id"])
        return ordered_ids

    # ------------------------------------------------------------------ #
    # Subgraph extraction
    # ------------------------------------------------------------------ #
    def extract_subgraph_by_nodes(
        self, node_names: List[str], max_path_length: int = 10
    ) -> Dict[str, Any]:
        key_nodes = [n for n in node_names if self._gs.get_node(n) is not None]
        all_paths: List[List[Dict[str, Any]]] = []
        for i, a in enumerate(key_nodes):
            for b in key_nodes[i + 1:]:
                paths = self.get_paths_between_nodes(a, b, max_step=max_path_length)
                all_paths.extend(paths)
        other_node_names: Set[str] = set()
        for path in all_paths:
            for step in path:
                other_node_names.add(step["from"])
                other_node_names.add(step["to"])
        other_nodes = [
            self.get_node_info(n)
            for n in other_node_names
            if self._gs.get_node(n) is not None and n not in key_nodes
        ]
        other_nodes = [n for n in other_nodes if n is not None]
        activities: List[Dict[str, Any]] = []
        for n in key_nodes:
            activities.extend(self.get_actions_related_to_entity(n))
        seen_ids: Set[str] = set()
        deduped_activities: List[Dict[str, Any]] = []
        for a in activities:
            if a["action_id"] not in seen_ids:
                seen_ids.add(a["action_id"])
                deduped_activities.append(a)
        return {
            "key_nodes": [self.get_node_info(n) for n in key_nodes],
            "other_nodes": other_nodes,
            "activities": deduped_activities,
            "paths": all_paths,
            "relationships": self._all_relationships_among(set(key_nodes) | other_node_names),
        }

    def _all_relationships_among(self, node_set: Set[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for nid in node_set:
            for oid, rtype, props in self._gs.get_neighbors(nid, direction="out"):
                if oid in node_set:
                    key = (nid, oid, rtype)
                    if key not in seen:
                        seen.add(key)
                        out.append({
                            "source": nid,
                            "target": oid,
                            "type": rtype,
                            "props": dict(props),
                        })
        return out

    def format_subgraph_json(self, subgraph: Dict[str, Any]) -> str:
        return json.dumps(subgraph, indent=2, default=str)

    def format_subgraph(self, subgraph: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("Key Nodes:")
        for n in subgraph.get("key_nodes", []):
            if n is None:
                continue
            lines.append(f"  - {n['label']} {n['name']}: {n['props'].get('info', '')}")
        lines.append("Other Nodes:")
        for n in subgraph.get("other_nodes", []):
            if n is None:
                continue
            lines.append(f"  - {n['label']} {n['name']}")
        lines.append("Activities:")
        for a in subgraph.get("activities", []):
            lines.append(f"  - {a['action_id']}: {a['props'].get('action_name', '')}")
        lines.append("Paths:")
        for path in subgraph.get("paths", []):
            steps = " → ".join(f"{s['from']} --{s['type']}--> {s['to']}" for s in path)
            lines.append(f"  - {steps}")
        return "\n".join(lines)

    def count_triples(self) -> int:
        total = 0
        for nid, _, _ in self._iter_nodes():
            total += len(self._gs.get_neighbors(nid, direction="out"))
        return total

    def all_node_names(self) -> List[str]:
        """Return every node name stored under the CLiViS slot label."""
        out: List[str] = []
        for nid, labels, _ in self._iter_nodes():
            if self.SLOT.value in labels:
                out.append(nid)
        return out

    # ------------------------------------------------------------------ #
    # Stats / introspection (for unimem integration)
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        per_label: Dict[str, int] = {}
        n_nodes = 0
        for _, labels, _ in self._iter_nodes():
            for l in labels:
                if l == self.SLOT.value:
                    continue
                per_label[l] = per_label.get(l, 0) + 1
            n_nodes += 1
        return {
            "count": n_nodes,
            "n_relationships": self.count_triples(),
            "n_actions": per_label.get("Activity", 0),
            "per_label": per_label,
        }

    def clear(self) -> None:
        # Drop every node carrying the slot label.
        self._gs.query(
            f"MATCH (n:{self.SLOT.value}) DETACH DELETE n"
        )

    # ------------------------------------------------------------------ #
    # MemoryModule contract — dispatches on entry.metadata["kind"]
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        kind = entry.metadata.get("kind", "node")
        if kind in {"person", "object", "area", "activity"}:
            label_map = {
                "person": "Person", "object": "Object",
                "area": "Area", "activity": "Activity",
            }
            kind = "node"
            label = entry.metadata.get("label", label_map[entry.metadata.get("kind")])
            entry.metadata["label"] = label
        if kind == "node":
            label = entry.metadata.get("label", "Object")
            name = entry.metadata.get("name", entry.entry_id)
            props = entry.metadata.get("props", {})
            if entry.text and "info" not in props:
                props.setdefault("info", entry.text)
            return self.add_update_node(name, label, props)
        if kind == "relation":
            return self.add_relation(
                entry.metadata["source"],
                entry.metadata["target"],
                entry.metadata["type"],
                entry.metadata.get("info", entry.text),
                entry.metadata.get("start_time", ""),
                entry.metadata.get("end_time"),
            )
        if kind == "action":
            return bool(self.add_action(
                action_name=entry.metadata["action_name"],
                action_info=entry.text,
                time_range=entry.metadata["time_range"],
                node_agent_name=entry.metadata.get("agent"),
                node_patient_name=entry.metadata.get("patient"),
                node_instrument_name=entry.metadata.get("instrument"),
                node_source_name=entry.metadata.get("source"),
                node_target_name=entry.metadata.get("target"),
                prev_action_id=entry.metadata.get("prev_action_id"),
            ))
        return False

    def read(self, query: Query) -> QueryResult:
        if query.semantic:
            triples: List[Tuple[str, str, Any]] = []
            for key in query.semantic:
                for src, _, _ in self._iter_nodes():
                    for oid, rtype, _ in self._gs.get_neighbors(src, key, "out"):
                        triples.append((src, rtype, oid))
            seen: Set[Tuple[str, str, str]] = set()
            entries: List[MemoryEntry] = []
            for s, p, o in triples:
                sig = (s, p, str(o))
                if sig in seen:
                    continue
                seen.add(sig)
                entries.append(MemoryEntry(
                    entry_id=f"rel-{len(entries)}",
                    text=f"({s}, {p}, {o})",
                    semantic_keys=[s, p, str(o)],
                    source_slot=self.SLOT.value,
                ))
            if query.top_k is not None:
                entries = entries[: query.top_k]
            return QueryResult(entries=entries, source_slot=self.SLOT.value)
        # No semantic filter → return all nodes
        entries = [
            MemoryEntry(
                entry_id=f"node-{nid}",
                text=f"{[l for l in labels if l != self.SLOT.value][:1]} {nid}: {props.get('info', '')}",
                semantic_keys=[nid],
                source_slot=self.SLOT.value,
                metadata={"name": nid, "labels": list(labels)},
            )
            for nid, labels, props in self._iter_nodes()
            if self.SLOT.value in labels
        ]
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot=self.SLOT.value)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _iter_nodes(self) -> List[Tuple[str, List[str], Dict[str, Any]]]:
        return self._gs._iter_nodes()  # noqa: SLF001 — same-package access

    @staticmethod
    def _compose_action_id(action_name: str, time_range: str) -> str:
        h = hashlib.sha1(
            f"{action_name}|{time_range}".encode("utf-8")
        ).hexdigest()[:8]
        return f"action-{action_name}-{h}"


__all__ = ["RelationGraph", "NodeLabels"]
