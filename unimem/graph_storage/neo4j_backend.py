"""Neo4j backend for :class:`~unimem.graph_storage.base.GraphStorage`.

Maps the unimem node-label / edge-type conventions onto Neo4j's native
property-graph model:

* **Node identity**: a string ``node_id`` property with a unique constraint
  on the ``Entity`` label. All nodes carry the ``Entity`` label so MERGE
  works regardless of which additional labels they also get.
* **Multiple labels**: ``Entity`` is the merge key; all other labels are
  added via APOC ``create.addLabels`` (Cypher itself can't parameterise
  labels).
* **Edges**: typed relationships with primitive properties. Neo4j cannot
  store nested dicts on edges, so :func:`_encode_value` JSON-encodes them
  with a sentinel prefix; :func:`_decode_value` reverses this on read.
* **Cypher passthrough**: :meth:`query` forwards the query string to Neo4j
  unchanged. Returned nodes are normalised to dicts that mirror
  :meth:`InMemoryGraphStorage._node_to_row` (so callers can use
  ``row["n"]["text"]`` and ``row["n"]["labels"]`` uniformly across backends).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import GraphStorage


# Sentinel prefix used to mark JSON-encoded property values. Chosen to be
# extremely unlikely to occur as a natural property value.
_JSON_SENTINEL = "\x00json\x00"


class Neo4jGraphStorage(GraphStorage):
    """Neo4j-backed implementation of :class:`GraphStorage`."""

    DEFAULT_URI = "bolt://localhost:7687"
    DEFAULT_USER = "neo4j"
    DEFAULT_PASSWORD = "password"
    DEFAULT_DATABASE = "neo4j"

    # Unimem stores node identity in this property. A unique constraint
    # over the catch-all ``Entity`` label makes MERGE fast & safe.
    ID_PROP = "node_id"
    MERGE_LABEL = "Entity"

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        database: str = DEFAULT_DATABASE,
        clear_on_init: bool = False,
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._driver.verify_connectivity()
        self._init_constraints()
        if clear_on_init:
            self._clear_all()

    # ------------------------------------------------------------------ #
    # Low-level CRUD
    # ------------------------------------------------------------------ #
    def add_node(
        self, node_id: str, labels: List[str], properties: Dict[str, Any]
    ) -> bool:
        """MERGE a node by ``node_id`` and merge labels + properties.

        Labels are added via APOC (Cypher itself cannot parameterise them).
        """
        extra_labels = [lab for lab in labels if lab]
        # De-duplicate while preserving order
        seen: set = set()
        extra_labels = [
            l for l in extra_labels
            if not (l in seen or seen.add(l))
        ]
        encoded = {k: _encode_value(v) for k, v in properties.items()}
        encoded[self.ID_PROP] = node_id

        cypher = (
            f"MERGE (n:{self.MERGE_LABEL} {{{self.ID_PROP}: $node_id}}) "
            "SET n += $props "
            "WITH n "
            "CALL apoc.create.addLabels(n, $labels) YIELD node "
            "RETURN node"
        )
        with self._session() as s:
            s.run(
                cypher, node_id=node_id, props=encoded, labels=extra_labels
            ).consume()
        return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        safe_type = self._sanitise_rel_type(rel_type)
        encoded = {
            k: _encode_value(v)
            for k, v in (properties or {}).items()
        }
        cypher = (
            "MATCH (s {" + self.ID_PROP + ": $src}), (t {" + self.ID_PROP + ": $tgt}) "
            f"MERGE (s)-[r:{safe_type}]->(t) "
            "SET r += $props "
            "RETURN type(r) AS t"
        )
        with self._session() as s:
            row = s.run(
                cypher, src=source_id, tgt=target_id, props=encoded
            ).single()
        return row is not None

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (n {" + self.ID_PROP + ": $id}) "
            "RETURN n, labels(n) AS labels"
        )
        with self._session() as s:
            row = s.run(cypher, id=node_id).single()
        if row is None:
            return None
        node_props = dict(row["n"])
        # Strip the merge label from the externally-visible label list
        all_labels = [
            l for l in row["labels"] if l != self.MERGE_LABEL
        ]
        decoded = {k: _decode_value(v) for k, v in node_props.items()}
        return {"labels": all_labels, "properties": decoded}

    def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        if direction not in ("out", "in", "both"):
            raise ValueError(
                f"direction must be 'out' / 'in' / 'both', got {direction!r}"
            )
        rel_filter = (
            f":`{self._sanitise_rel_type(rel_type)}`" if rel_type else ""
        )
        if direction == "out":
            pattern = f"(n)-[r{rel_filter}]->(m)"
        elif direction == "in":
            pattern = f"(n)<-[r{rel_filter}]-(m)"
        else:
            pattern = f"(n)-[r{rel_filter}]-(m)"
        cypher = (
            "MATCH " + pattern + " "
            "WHERE n." + self.ID_PROP + " = $id "
            "RETURN m." + self.ID_PROP + " AS other, "
            "type(r) AS rtype, properties(r) AS rprops"
        )
        with self._session() as s:
            rows = s.run(cypher, id=node_id).data()
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        for r in rows:
            if r.get("other") is None:
                continue
            decoded = {
                k: _decode_value(v) for k, v in dict(r["rprops"]).items()
            }
            out.append((r["other"], r["rtype"], decoded))
        return out

    def delete_node(self, node_id: str) -> int:
        cypher = (
            "MATCH (n {" + self.ID_PROP + ": $id}) "
            "DETACH DELETE n "
            "RETURN count(n) AS c"
        )
        with self._session() as s:
            row = s.run(cypher, id=node_id).single()
        return int(row["c"]) if row else 0

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: Optional[str] = None,
    ) -> int:
        rel_filter = (
            f":`{self._sanitise_rel_type(rel_type)}`" if rel_type else ""
        )
        cypher = (
            "MATCH (s {" + self.ID_PROP + ": $src})-[r" + rel_filter + "]->"
            "(t {" + self.ID_PROP + ": $tgt}) "
            "DELETE r "
            "RETURN count(r) AS c"
        )
        with self._session() as s:
            row = s.run(cypher, src=source_id, tgt=target_id).single()
        return int(row["c"]) if row else 0

    def query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query and return rows as plain dicts.

        Neo4j ``Node`` objects are normalised so that the returned dict
        mirrors :meth:`InMemoryGraphStorage._node_to_row`: it contains
        ``labels``, ``properties``, ``node_id``, ``id``, plus flattened
        property keys for direct access (e.g. ``row["n"]["text"]``).

        Neo4j ``Relationship`` objects are normalised to
        ``{"type", "source_id", "target_id", "properties"}``.
        """
        with self._session() as s:
            records = list(s.run(query, parameters or {}))
        out: List[Dict[str, Any]] = []
        for rec in records:
            row: Dict[str, Any] = {}
            for key in rec.keys():
                row[key] = self._normalise_value(rec[key])
            out.append(row)
        return out

    def bfs(
        self,
        start_id: str,
        rel_types: Optional[Sequence[str]] = None,
        max_depth: int = 100,
    ) -> List[str]:
        if rel_types:
            rel_filter = "|".join(
                f"`{self._sanitise_rel_type(t)}`" for t in rel_types
            )
        else:
            rel_filter = ""
        cypher = (
            "MATCH (start {" + self.ID_PROP + ": $id}) "
            "CALL apoc.path.expandConfig(start, { "
            f"  relationshipFilter: '>{rel_filter}', "
            "  uniqueness: 'NODE_GLOBAL', "
            "  maxLevel: $max_depth "
            "}) "
            "YIELD path "
            "UNWIND [n IN nodes(path) | n." + self.ID_PROP + "] AS nid "
            "RETURN DISTINCT nid"
        )
        with self._session() as s:
            rows = s.run(
                cypher, id=start_id, max_depth=int(max_depth)
            ).data()
        return [r["nid"] for r in rows if r.get("nid") and r["nid"] != start_id]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jGraphStorage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _session(self):
        return self._driver.session(database=self._database)

    def _init_constraints(self) -> None:
        cypher = (
            "CREATE CONSTRAINT node_id_unique IF NOT EXISTS "
            f"FOR (n:{self.MERGE_LABEL}) REQUIRE n." + self.ID_PROP
            + " IS UNIQUE"
        )
        with self._session() as s:
            s.run(cypher).consume()

    def _clear_all(self) -> None:
        with self._session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()

    @staticmethod
    def _sanitise_rel_type(rtype: str) -> str:
        rtype = str(rtype)
        if all(c.isalnum() or c == "_" for c in rtype):
            return rtype
        return f"`{rtype.replace('`', '``')}`"

    def _normalise_value(self, v: Any) -> Any:
        """Convert Neo4j driver objects to plain Python values.

        ``Node`` and ``Relationship`` from ``neo4j`` are normalised to
        plain dicts. JSON-encoded property values are decoded.
        """
        # Neo4j Node — duck-typed (has labels + items)
        if self._is_node(v):
            labels = [l for l in v.labels if l != self.MERGE_LABEL]
            props = {k: _decode_value(val) for k, val in dict(v).items()}
            flat = dict(props)
            flat["labels"] = labels
            flat["node_id"] = props.get(self.ID_PROP)
            flat["id"] = props.get(self.ID_PROP)
            flat["properties"] = props
            return flat
        # Neo4j Relationship
        if self._is_relationship(v):
            return {
                "type": v.type,
                "source_id": dict(v.start_node).get(self.ID_PROP)
                if hasattr(v, "start_node") and v.start_node is not None
                else None,
                "target_id": dict(v.end_node).get(self.ID_PROP)
                if hasattr(v, "end_node") and v.end_node is not None
                else None,
                "properties": {
                    k: _decode_value(val) for k, val in dict(v).items()
                },
            }
        if isinstance(v, list):
            return [self._normalise_value(x) for x in v]
        return v

    @staticmethod
    def _is_node(v: Any) -> bool:
        # Neo4j Node: has .labels (frozenset) and .items() but isn't a dict
        if isinstance(v, dict):
            return False
        return hasattr(v, "labels") and hasattr(v, "items") and hasattr(v, "id")

    @staticmethod
    def _is_relationship(v: Any) -> bool:
        if isinstance(v, dict):
            return False
        return (
            hasattr(v, "type")
            and hasattr(v, "items")
            and hasattr(v, "start_node")
            and hasattr(v, "end_node")
            and callable(getattr(v, "type", None)) is False  # type is a property, not method
        )

    # ------------------------------------------------------------------ #
    # Override _iter_nodes to use our `node_id` property as the id
    # (default impl uses Neo4j's deprecated internal id()).
    # ------------------------------------------------------------------ #
    def _iter_nodes(self) -> List[Tuple[str, List[str], Dict[str, Any]]]:
        rows = self.query(
            "MATCH (n) RETURN n." + self.ID_PROP + " AS nid, "
            "labels(n) AS labels, n"
        )
        out: List[Tuple[str, List[str], Dict[str, Any]]] = []
        for row in rows:
            node_id = row.get("nid")
            if node_id is None:
                continue
            # Strip the merge label from the externally-visible label list
            visible_labels = [
                l for l in (row.get("labels") or []) if l != self.MERGE_LABEL
            ]
            props = dict(row.get("n", {}).get("properties", {}))
            out.append((str(node_id), visible_labels, props))
        return out


def _encode_value(v: Any) -> Any:
    """Convert a value to a Neo4j-storable primitive.

    Neo4j property values can only be primitives or arrays of primitives.
    Anything else (dict / nested list / None) is JSON-encoded with a
    sentinel prefix.
    """
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        # Allow only if every element is a primitive; otherwise encode whole list
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
            return list(v)
        return _JSON_SENTINEL + json.dumps(list(v), default=str)
    if isinstance(v, dict):
        return _JSON_SENTINEL + json.dumps(v, default=str)
    # Fallback: any other object → JSON
    return _JSON_SENTINEL + json.dumps(v, default=str)


def _decode_value(v: Any) -> Any:
    """Reverse of :func:`_encode_value`."""
    if isinstance(v, str) and v.startswith(_JSON_SENTINEL):
        try:
            return json.loads(v[len(_JSON_SENTINEL):])
        except (json.JSONDecodeError, ValueError):
            return v
    return v


__all__ = ["Neo4jGraphStorage"]
