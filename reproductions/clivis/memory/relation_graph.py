"""CLiViS ``RelationGraph`` — pure-Python property graph (no Neo4j).

Reproduces ``reproduce/CLiViS/clivis/graph/relation_graph.py`` without the
external Neo4j dependency. The original issues Cypher queries against a
live database; here we use in-memory dicts to model the same property
graph (nodes with labels + properties, typed relationships with properties).

Covers the original's public API surface:

* Node ops: ``add_person``, ``add_update_objects``, ``add_area``,
  ``add_update_node``, ``get_node_info``
* Relationship ops: ``add_relation``, ``get_relation_info``,
  ``get_relations_of_node``, ``get_paths_between_nodes``
* Action ops: ``add_action``, ``get_action_with_relations``,
  ``get_action_chain``, ``get_actions_related_to_entity``,
  ``get_actions_in_period``, ``get_all_actions_in_time_range``
* Subgraph extraction: ``extract_subgraph_by_nodes``, ``format_subgraph``,
  ``format_subgraph_json``, ``count_triples``

Implements both :class:`~unimem.core.slot_abc.SceneGraphMemoryABC` and
:class:`~unimem.core.slot_abc.SemanticMemoryABC` so the unimem graph can
treat this module as a scene graph (tree-style traversal) *and* as a
semantic fact store (triple queries).
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import Query, QueryResult
from unimem.core.slot_abc import SceneGraphMemoryABC, SemanticMemoryABC


# --------------------------------------------------------------------------- #
# Node labels (mirror the original Neo4j labels)
# --------------------------------------------------------------------------- #
class NodeLabels(Enum):
    PERSON = "Person"
    OBJECT = "Object"
    ACTIVITY = "Activity"
    AREA = "Area"

    @classmethod
    def from_value(cls, value: str) -> "NodeLabels":
        for m in cls:
            if m.value == value or m.name == value:
                return m
        # Tolerate alternate casings the LLM may produce
        norm = value.strip().lower()
        for m in cls:
            if m.value.lower() == norm or m.name.lower() == norm:
                return m
        raise KeyError(f"Unknown NodeLabels value: {value!r}")


# Standard typed relationship names used by ``add_action``
ACTION_REL_TYPES = {
    "agent": "PERFORMS",
    "patient": "AFFECTS",
    "instrument": "USES",
    "source": "FROM",
    "target": "TO",
}


class _Node:
    __slots__ = ("name", "label", "props")

    def __init__(self, name: str, label: str, props: Optional[Dict[str, Any]] = None):
        self.name = name
        self.label = label
        self.props = dict(props) if props else {}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label, "props": dict(self.props)}


class _Rel:
    __slots__ = ("source", "target", "type", "props")

    def __init__(self, source: str, target: str, rel_type: str, props: Optional[Dict[str, Any]] = None):
        self.source = source
        self.target = target
        self.type = rel_type
        self.props = dict(props) if props else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "props": dict(self.props),
        }


# --------------------------------------------------------------------------- #
# RelationGraph
# --------------------------------------------------------------------------- #
class RelationGraph(SceneGraphMemoryABC, SemanticMemoryABC):
    """In-memory property graph mirroring CLiViS's Neo4j schema.

    Nodes: ``Person`` / ``Object`` / ``Area`` / ``Activity``.
    Relationships: arbitrary string types (PERFORMS, AFFECTS, USES, FROM, TO,
    NEXT_ACTION, plus any freeform type the LLM extracts).
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, _Node] = {}
        self._out_rels: Dict[str, List[_Rel]] = {}
        self._in_rels: Dict[str, List[_Rel]] = {}
        self._action_chain: Dict[str, Optional[str]] = {}  # action_id -> next_action_id
        self._action_prev: Dict[str, Optional[str]] = {}   # action_id -> prev_action_id

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #
    def add_update_node(
        self, node_name: str, node_label: str, attr_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not node_name:
            return False
        # Validate label
        try:
            NodeLabels.from_value(node_label)
        except KeyError:
            return False
        if node_name in self._nodes:
            self._nodes[node_name].props.update(attr_dict or {})
        else:
            self._nodes[node_name] = _Node(node_name, node_label, attr_dict)
            self._out_rels[node_name] = []
            self._in_rels[node_name] = []
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
        node = self._nodes.get(node_name)
        if node is None:
            return None
        if node_label is not None and node.label != NodeLabels.from_value(node_label).value:
            return None
        return node.to_dict()

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
        if node_a_name not in self._nodes or node_b_name not in self._nodes:
            return False
        rel = _Rel(
            source=node_a_name,
            target=node_b_name,
            rel_type=relation_type,
            props={
                "info": relation_info,
                "start_time": start_time,
                "end_time": end_time if end_time is not None else "",
            },
        )
        self._out_rels[node_a_name].append(rel)
        self._in_rels[node_b_name].append(rel)
        return True

    def get_relation_info(self, relation_name: str) -> Optional[Dict[str, Any]]:
        for src, rels in self._out_rels.items():
            for r in rels:
                if r.type == relation_name:
                    return r.to_dict()
        return None

    def get_relations_of_node(
        self, node_name: str, node_label: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        if node_name not in self._nodes:
            return {"outgoing": [], "incoming": []}
        outgoing = [
            {"type": r.type, "endpoint": r.target, "props": dict(r.props)}
            for r in self._out_rels.get(node_name, [])
        ]
        incoming = [
            {"type": r.type, "endpoint": r.source, "props": dict(r.props)}
            for r in self._in_rels.get(node_name, [])
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
        if node_a_name not in self._nodes or node_b_name not in self._nodes:
            return []
        results: List[List[Dict[str, Any]]] = []

        def neighbours(n: str) -> List[Tuple[str, _Rel]]:
            nb = [(r.target, r) for r in self._out_rels.get(n, [])]
            if dual_direction:
                nb += [(r.source, r) for r in self._in_rels.get(n, [])]
            return nb

        # DFS
        stack: List[Tuple[str, List[str], List[Dict[str, Any]]]] = [
            (node_a_name, [node_a_name], [])
        ]
        while stack:
            current, path, edge_path = stack.pop()
            if len(edge_path) >= max_step:
                continue
            for nb, rel in neighbours(current):
                if nb in path:
                    continue
                step = list(edge_path) + [{
                    "from": rel.source, "to": rel.target, "type": rel.type,
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
        node_agent_name: str,
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
        # Wire role relationships
        role_pairs = [
            (node_agent_name, "PERFORMS"),
            (node_patient_name, "AFFECTS"),
            (node_instrument_name, "USES"),
            (node_source_name, "FROM"),
            (node_target_name, "TO"),
        ]
        for endpoint, rel_type in role_pairs:
            if endpoint and endpoint in self._nodes:
                # action → entity (action performs agent, action affects patient, ...)
                self.add_relation(action_id, endpoint, rel_type, "", time_range, "")
        # Chain: prev → me
        if prev_action_id is not None and prev_action_id in self._nodes:
            self.add_relation(prev_action_id, action_id, "NEXT_ACTION", "", time_range, "")
            self._action_prev[action_id] = prev_action_id
            self._action_chain[prev_action_id] = action_id
        self._action_prev.setdefault(action_id, None)
        self._action_chain.setdefault(action_id, None)
        return action_id

    def get_action_with_relations(self, action_id: str) -> Optional[Dict[str, Any]]:
        node = self._nodes.get(action_id)
        if node is None or node.label != "Activity":
            return None
        rels = self.get_relations_of_node(action_id)
        return {
            "action_id": action_id,
            "props": dict(node.props),
            "relations": rels,
            "prev_action_id": self._action_prev.get(action_id),
            "next_action_id": self._action_chain.get(action_id),
        }

    def get_action_chain(self, start_action_id: Optional[str] = None) -> List[str]:
        """Return the chain of actions starting from the given id, or all roots."""
        if start_action_id is not None:
            cursor: Optional[str] = start_action_id
        else:
            # All actions with no prev
            roots = [aid for aid, prev in self._action_prev.items() if prev is None]
            cursor = roots[0] if roots else None
        chain: List[str] = []
        seen: Set[str] = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            cursor = self._action_chain.get(cursor)
        return chain

    def get_actions_related_to_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        if entity_name not in self._nodes:
            return []
        out = []
        for r in self._in_rels.get(entity_name, []):
            if r.source in self._nodes and self._nodes[r.source].label == "Activity":
                out.append(self.get_action_with_relations(r.source))
        return [a for a in out if a is not None]

    def get_actions_in_period(
        self, time_range: str, agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return actions whose ``time_range`` overlaps the given one."""
        from .navigation_graph import _parse_range_seconds, _ranges_overlap

        target = _parse_range_seconds(time_range)
        actions = []
        for aid, node in self._nodes.items():
            if node.label != "Activity":
                continue
            ar = _parse_range_seconds(node.props.get("time_range", ""))
            if target is None or ar is None or _ranges_overlap(target, ar):
                if agent_name is None or agent_name in [
                    r.target for r in self._out_rels.get(aid, []) if r.type == "PERFORMS"
                ]:
                    actions.append(self.get_action_with_relations(aid))
        return actions

    def get_all_actions_in_time_range(self, time_range: str) -> List[str]:
        actions = self.get_actions_in_period(time_range)
        chain = self.get_action_chain()
        ordered_ids = [aid for aid in chain if any(a["action_id"] == aid for a in actions)]
        # Include any actions not on the main chain
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
        """Build a subgraph around the given key nodes."""
        key_nodes = [n for n in node_names if n in self._nodes]
        all_paths = []
        for i, a in enumerate(key_names := key_nodes):
            for b in key_nodes[i + 1:]:
                paths = self.get_paths_between_nodes(a, b, max_step=max_path_length)
                all_paths.extend(paths)
        # Other nodes touched by paths
        other_node_names: Set[str] = set()
        for path in all_paths:
            for step in path:
                other_node_names.add(step["from"])
                other_node_names.add(step["to"])
        other_nodes = [
            self._nodes[n].to_dict()
            for n in other_node_names
            if n in self._nodes and n not in key_nodes
        ]
        # Activities referencing any key node
        activities = []
        for n in key_nodes:
            activities.extend(self.get_actions_related_to_entity(n))
        # Dedup activities by action_id
        seen_ids: Set[str] = set()
        deduped_activities = []
        for a in activities:
            if a["action_id"] not in seen_ids:
                seen_ids.add(a["action_id"])
                deduped_activities.append(a)
        return {
            "key_nodes": [self._nodes[n].to_dict() for n in key_nodes],
            "other_nodes": other_nodes,
            "activities": deduped_activities,
            "paths": all_paths,
            "relationships": self._all_relationships_among(set(key_nodes) | other_node_names),
        }

    def _all_relationships_among(self, node_set: Set[str]) -> List[Dict[str, Any]]:
        out = []
        seen: Set[Tuple[str, str, str]] = set()
        for src in node_set:
            for r in self._out_rels.get(src, []):
                if r.target in node_set:
                    key = (r.source, r.target, r.type)
                    if key not in seen:
                        seen.add(key)
                        out.append(r.to_dict())
        return out

    def format_subgraph_json(self, subgraph: Dict[str, Any]) -> str:
        return json.dumps(subgraph, indent=2, default=str)

    def format_subgraph(self, subgraph: Dict[str, Any]) -> str:
        lines = []
        lines.append("Key Nodes:")
        for n in subgraph.get("key_nodes", []):
            lines.append(f"  - {n['label']} {n['name']}: {n['props'].get('info', '')}")
        lines.append("Other Nodes:")
        for n in subgraph.get("other_nodes", []):
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
        return sum(len(rels) for rels in self._out_rels.values())

    # ------------------------------------------------------------------ #
    # SceneGraphMemoryABC
    # ------------------------------------------------------------------ #
    def add_object(
        self, object_id: str, parent_id: Optional[str] = None, **attrs: Any
    ) -> bool:
        """Treat ``add_object`` as a generic node upsert.

        ``parent_id`` is recorded as a SUBSUMES-style relationship
        (type=``"CONTAINS"`` from parent to child), mirroring the original's
        ``area CONTAINS object`` relations.
        """
        label = attrs.pop("label", "Object")
        ok = self.add_update_node(object_id, label, attrs)
        if parent_id is not None and parent_id in self._nodes:
            self.add_relation(parent_id, object_id, "CONTAINS", "", "", "")
        return ok

    def get_children(self, parent_id: Optional[str]) -> List[str]:
        if parent_id is None:
            # Root nodes: those with no incoming CONTAINS
            return [
                name for name, node in self._nodes.items()
                if not any(r.type == "CONTAINS" for r in self._in_rels.get(name, []))
            ]
        return [
            r.target for r in self._out_rels.get(parent_id, []) if r.type == "CONTAINS"
        ]

    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        info = self.get_node_info(object_id)
        if info is None:
            return None
        return info["props"]

    # ------------------------------------------------------------------ #
    # SemanticMemoryABC (triple view)
    # ------------------------------------------------------------------ #
    def add_fact(self, subject: str, predicate: str, obj: Any) -> bool:
        # Subject must be a node; object is treated as a freeform value
        # encoded as a relationship to a synthetic node.
        if subject not in self._nodes:
            return False
        obj_name = str(obj)
        if obj_name not in self._nodes:
            # Create a placeholder node so the relationship has somewhere to land
            self.add_update_node(obj_name, "Object", {"value": obj})
        self.add_relation(subject, obj_name, predicate, "", "", "")
        return True

    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Any = None,
    ) -> List[Tuple[str, str, Any]]:
        out: List[Tuple[str, str, Any]] = []
        for src, rels in self._out_rels.items():
            if subject is not None and src != subject:
                continue
            for r in rels:
                if predicate is not None and r.type != predicate:
                    continue
                if obj is not None and r.target != str(obj):
                    continue
                out.append((src, r.type, r.target))
        return out

    # ------------------------------------------------------------------ #
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Generic write from a structured ``MemoryEntry``.

        Accepted shapes:

        * ``metadata['kind'] == 'node'`` — add node (uses ``metadata['label']``,
          ``metadata['props']``, ``metadata['name']`` or ``entry_id``).
        * ``metadata['kind'] in {'person', 'object', 'area', 'activity'}`` —
          shortcut for ``kind=node`` with the corresponding label.
        * ``metadata['kind'] == 'relation'`` — add relation (source/target/type
          + props).
        * ``metadata['kind'] == 'action'`` — add_action(...).
        * Fallback: treat as a generic node add with text=info.
        """
        kind = entry.metadata.get("kind", "node")
        # Normalise slot-specific kinds to "node" with the right label
        if kind in {"person", "object", "area", "activity"}:
            label_map = {
                "person": "Person", "object": "Object",
                "area": "Area", "activity": "Activity",
            }
            entry = MemoryEntry(
                entry_id=entry.entry_id,
                text=entry.text,
                metadata={
                    **entry.metadata,
                    "kind": "node",
                    "label": entry.metadata.get("label", label_map[kind]),
                },
                payload=entry.payload,
                semantic_keys=list(entry.semantic_keys),
                spatial_keys=list(entry.spatial_keys),
                temporal_keys=list(entry.temporal_keys),
                source_slot=entry.source_slot,
            )
            kind = "node"
        if kind == "node":
            label = entry.metadata.get("label", "Object")
            name = entry.metadata.get("name", entry.entry_id)
            props = entry.metadata.get("props", {})
            if not entry.text and "info" not in props:
                props["info"] = entry.text
            elif entry.text and "info" not in props:
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
        """Retrieve facts or nodes matching the query.

        With ``semantic_keys`` present, returns matching triples
        (subject, predicate, object) wrapped as MemoryEntry text. Otherwise
        returns all nodes (subject to ``top_k``).
        """
        if query.semantic:
            # Treat each semantic key as a predicate to match
            triples: List[Tuple[str, str, Any]] = []
            for key in query.semantic:
                triples.extend(self.query_facts(predicate=key))
            # Also include triples whose subject or object matches a key
            for key in query.semantic:
                triples.extend(self.query_facts(subject=key))
            # Dedup
            seen = set()
            entries = []
            for s, p, o in triples:
                sig = (s, p, str(o))
                if sig in seen:
                    continue
                seen.add(sig)
                entries.append(MemoryEntry(
                    entry_id=f"rel-{len(entries)}",
                    text=f"({s}, {p}, {o})",
                    semantic_keys=[s, p, str(o)],
                    source_slot="semantic",
                    metadata={"subject": s, "predicate": p, "object": o},
                ))
            if query.top_k is not None:
                entries = entries[: query.top_k]
            return QueryResult(entries=entries, source_slot="semantic")
        # No semantic filter → return all nodes
        entries = [
            MemoryEntry(
                entry_id=f"node-{name}",
                text=f"{node.label} {name}: {node.props.get('info', '')}",
                semantic_keys=[name, node.label],
                source_slot="scene_graph",
                metadata={"name": name, "label": node.label, "props": dict(node.props)},
            )
            for name, node in self._nodes.items()
        ]
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot="scene_graph")

    def clear(self) -> None:
        self._nodes.clear()
        self._out_rels.clear()
        self._in_rels.clear()
        self._action_chain.clear()
        self._action_prev.clear()

    def stats(self) -> Dict[str, Any]:
        per_label = {}
        for n in self._nodes.values():
            per_label[n.label] = per_label.get(n.label, 0) + 1
        return {
            "count": len(self._nodes),
            "n_relationships": self.count_triples(),
            "n_actions": per_label.get("Activity", 0),
            "per_label": per_label,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_action_id(action_name: str, time_range: str) -> str:
        # Deterministic id; the original uses a hash. We use a readable slug.
        import hashlib
        h = hashlib.sha1(
            f"{action_name}|{time_range}".encode("utf-8")
        ).hexdigest()[:8]
        return f"action-{action_name}-{h}"


__all__ = ["RelationGraph", "NodeLabels"]
