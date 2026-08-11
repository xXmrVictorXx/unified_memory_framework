"""``InMemoryGraphStorage`` — pure-stdlib fallback backend.

A property graph stored entirely in process memory. Supports a small subset
of Cypher-like queries sufficient for unimem's needs (MATCH by label,
WHERE on properties, RETURN nodes / relationships / aggregations).

This backend is the default when no external graph DB is configured. It is
also the backend used by every unimem unit test.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import GraphStorage


class _Node:
    __slots__ = ("node_id", "labels", "properties")

    def __init__(
        self,
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> None:
        self.node_id = node_id
        self.labels = list(labels)
        self.properties = dict(properties)


class _Edge:
    __slots__ = ("source_id", "target_id", "rel_type", "properties")

    def __init__(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict[str, Any],
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.rel_type = rel_type
        self.properties = dict(properties)


class InMemoryGraphStorage(GraphStorage):
    """Adjacency-list property graph in memory.

    Nodes and edges are stored in dicts keyed by id. Outgoing and incoming
    edge lists are kept per node for O(1) neighbour lookup.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, _Node] = {}
        # Adjacency: source -> list of edges
        self._out: Dict[str, List[_Edge]] = {}
        # Reverse adjacency: target -> list of edges
        self._in: Dict[str, List[_Edge]] = {}

    # ------------------------------------------------------------------ #
    # Low-level CRUD
    # ------------------------------------------------------------------ #
    def add_node(
        self, node_id: str, labels: List[str], properties: Dict[str, Any]
    ) -> bool:
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = _Node(node_id, list(labels), dict(properties))
            self._out.setdefault(node_id, [])
            self._in.setdefault(node_id, [])
        else:
            # Merge labels (preserve order, dedupe)
            for lab in labels:
                if lab not in existing.labels:
                    existing.labels.append(lab)
            existing.properties.update(properties)
        return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if source_id not in self._nodes or target_id not in self._nodes:
            return False
        edge = _Edge(source_id, target_id, rel_type, properties or {})
        # MERGE semantics: if an edge with same (s, t, type) exists, update props.
        for existing in self._out[source_id]:
            if (
                existing.target_id == target_id
                and existing.rel_type == rel_type
            ):
                existing.properties.update(properties or {})
                return True
        self._out[source_id].append(edge)
        self._in[target_id].append(edge)
        return True

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        return {"labels": list(node.labels), "properties": dict(node.properties)}

    def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be 'out' / 'in' / 'both', got {direction!r}")
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        if direction in ("out", "both"):
            for e in self._out.get(node_id, []):
                if rel_type is not None and e.rel_type != rel_type:
                    continue
                out.append((e.target_id, e.rel_type, dict(e.properties)))
        if direction in ("in", "both"):
            for e in self._in.get(node_id, []):
                if rel_type is not None and e.rel_type != rel_type:
                    continue
                out.append((e.source_id, e.rel_type, dict(e.properties)))
        return out

    def delete_node(self, node_id: str) -> int:
        if node_id not in self._nodes:
            return 0
        # Remove edges referencing this node
        for e in list(self._out.get(node_id, [])):
            self._drop_edge(e)
        for e in list(self._in.get(node_id, [])):
            self._drop_edge(e)
        del self._nodes[node_id]
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)
        return 1

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: Optional[str] = None,
    ) -> int:
        count = 0
        for e in list(self._out.get(source_id, [])):
            if e.target_id != target_id:
                continue
            if rel_type is not None and e.rel_type != rel_type:
                continue
            self._drop_edge(e)
            count += 1
        return count

    def bfs(
        self,
        start_id: str,
        rel_types: Optional[Sequence[str]] = None,
        max_depth: int = 100,
    ) -> List[str]:
        if start_id not in self._nodes:
            return []
        rel_set = set(rel_types) if rel_types is not None else None
        visited = {start_id}
        dq = deque([(start_id, 0)])
        out: List[str] = []
        while dq:
            nid, depth = dq.popleft()
            if depth >= max_depth:
                continue
            for e in self._out.get(nid, []):
                if rel_set is not None and e.rel_type not in rel_set:
                    continue
                if e.target_id in visited:
                    continue
                visited.add(e.target_id)
                out.append(e.target_id)
                dq.append((e.target_id, depth + 1))
        return out

    # ------------------------------------------------------------------ #
    # Cypher-ish query
    # ------------------------------------------------------------------ #
    _MATCH_PATTERN = re.compile(
        r"MATCH\s*\((?P<var>\w+)(?::(?P<label>\w+))?\)(?:\s*-\s*\[\s*(?P<evar>\w+)?(?::(?P<rtype>\w+))?\s*\]\s*->\s*\((?P<tvar>\w+)(?::(?P<tlabel>\w+))?\))?",
        re.IGNORECASE,
    )
    _WHERE_PATTERN = re.compile(
        r"WHERE\s+(?P<body>.+?)(?=\s+(?:RETURN|SET|DELETE|WITH|ORDER|LIMIT)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _RETURN_PATTERN = re.compile(
        r"RETURN\s+(?P<body>.+?)(?=\s+(?:ORDER|LIMIT)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _LIMIT_PATTERN = re.compile(r"LIMIT\s+(?P<n>\d+)", re.IGNORECASE)
    _ORDER_PATTERN = re.compile(
        r"ORDER\s+BY\s+(?P<body>.+?)(?=\s+(?:LIMIT|RETURN)|$)",
        re.IGNORECASE | re.DOTALL,
    )

    def query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        q = query.strip()
        # Simplistic DETACH DELETE / DELETE
        if q.upper().startswith("MATCH") and "DETACH DELETE" in q.upper():
            return self._exec_detach_delete(q, parameters)
        if q.upper().startswith("MATCH") and "DELETE" in q.upper() and "DETACH" not in q.upper():
            return self._exec_delete(q, parameters)
        if q.upper().startswith("MATCH"):
            return self._exec_match(q, parameters)
        if q.upper().startswith("MERGE") or q.upper().startswith("CREATE"):
            # Not implemented for the in-memory backend; callers should use
            # add_node / add_edge directly.
            return []
        return []

    def _exec_match(
        self, q: str, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        m = self._MATCH_PATTERN.search(q)
        if not m:
            return []
        var = m.group("var")
        label = m.group("label")
        evar = m.group("evar")
        rtype = m.group("rtype")
        tvar = m.group("tvar")
        tlabel = m.group("tlabel")

        # Build candidate bindings
        rows: List[Dict[str, Any]] = []
        if tvar is None:
            # Node-only pattern: iterate nodes that match the label filter.
            for nid, node in self._nodes.items():
                if label is not None and label not in node.labels:
                    continue
                rows.append({var: self._node_to_row(node)})
        else:
            # Pattern with edge: iterate edges
            for src_id, edges in self._out.items():
                src_node = self._nodes[src_id]
                if label is not None and label not in src_node.labels:
                    continue
                for e in edges:
                    if rtype is not None and e.rel_type != rtype:
                        continue
                    tgt_node = self._nodes.get(e.target_id)
                    if tgt_node is None:
                        continue
                    if tlabel is not None and tlabel not in tgt_node.labels:
                        continue
                    row = {
                        var: self._node_to_row(src_node),
                        tvar: self._node_to_row(tgt_node),
                    }
                    if evar:
                        row[evar] = {
                            "type": e.rel_type,
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "properties": dict(e.properties),
                        }
                    rows.append(row)

        # WHERE: very simple property equality on bound vars
        where = self._WHERE_PATTERN.search(q)
        if where:
            rows = self._apply_where(where.group("body"), rows, parameters)

        # RETURN
        ret = self._RETURN_PATTERN.search(q)
        if not ret:
            # Default: return matched nodes (as properties)
            return [{var: r[var] for var in r} for r in rows]
        return_body = ret.group("body").strip()

        # ORDER BY
        order = self._ORDER_PATTERN.search(q)
        if order:
            rows = self._apply_order_by(order.group("body").strip(), rows)

        # LIMIT
        limit = self._LIMIT_PATTERN.search(q)
        if limit:
            n = int(limit.group("n"))
            rows = rows[:n]

        # Aggregations: COUNT(*), COUNT(x)
        if return_body.upper().startswith("COUNT("):
            count_target = return_body[6:-1].strip()
            if count_target in ("*", ""):
                return [{"count": len(rows)}]
            # count of non-null
            return [{"count": sum(1 for r in rows if r.get(count_target) is not None)}]

        # Star return: return whole row
        if return_body == "*":
            return [dict(r) for r in rows]

        # Projection
        out: List[Dict[str, Any]] = []
        for r in rows:
            proj: Dict[str, Any] = {}
            for token in self._split_return(return_body):
                token = token.strip()
                if not token:
                    continue
                # Handle "var AS alias"
                as_match = re.match(
                    r"^(?P<expr>[\w.]+)\s+(?:AS\s+)?(?P<alias>\w+)$",
                    token,
                    re.IGNORECASE,
                )
                if as_match:
                    expr = as_match.group("expr")
                    alias = as_match.group("alias")
                else:
                    expr = token
                    alias = token
                proj[alias] = self._eval_return_expr(expr, r)
            out.append(proj)
        return out

    def _exec_detach_delete(self, q: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        m = self._MATCH_PATTERN.search(q)
        if not m:
            return []
        var = m.group("var")
        label = m.group("label")
        # Find the variable name being deleted
        del_match = re.search(r"DETACH\s+DELETE\s+(?P<var>\w+)", q, re.IGNORECASE)
        if not del_match:
            return []
        del_var = del_match.group("var")
        if del_var != var:
            return []
        deleted = 0
        for nid in list(self._nodes.keys()):
            node = self._nodes[nid]
            if label is not None and label not in node.labels:
                continue
            self.delete_node(nid)
            deleted += 1
        return [{"deleted": deleted}]

    def _exec_delete(self, q: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Minimal: MATCH (n:Label) DELETE n
        m = self._MATCH_PATTERN.search(q)
        if not m:
            return []
        var = m.group("var")
        label = m.group("label")
        deleted = 0
        for nid in list(self._nodes.keys()):
            node = self._nodes[nid]
            if label is not None and label not in node.labels:
                continue
            self.delete_node(nid)
            deleted += 1
        return [{"deleted": deleted}]

    # -- query helpers ---------------------------------------------------- #
    def _apply_where(
        self,
        body: str,
        rows: List[Dict[str, Any]],
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply WHERE expressions joined by AND.

        Supports:

        * ``var.prop = value``  (value is a literal or $param)
        * ``var.prop >= value`` / ``<=`` / ``>`` / ``<`` / ``<>``
        * ``label(var) = "Foo"`` (no-op; filtered at MATCH time)
        * ``var.prop IN [$param]`` (param resolves to a list)
        """
        clauses = re.split(r"\s+AND\s+", body.strip(), flags=re.IGNORECASE)
        out: List[Dict[str, Any]] = []
        for row in rows:
            keep = True
            for clause in clauses:
                if not self._eval_where_clause(clause.strip(), row, parameters):
                    keep = False
                    break
            if keep:
                out.append(row)
        return out

    def _eval_where_clause(
        self,
        clause: str,
        row: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> bool:
        # Try each comparison operator (longest first)
        for op in (">=", "<=", "<>", "!=", "==", "=", ">", "<"):
            # Split on first occurrence
            idx = clause.find(op)
            if idx > 0:
                lhs = clause[:idx].strip()
                rhs = clause[idx + len(op):].strip()
                lval = self._eval_return_expr(lhs, row)
                rval = self._eval_value(rhs, parameters)
                if op == "=" or op == "==":
                    return lval == rval
                if op == "<>" or op == "!=":
                    return lval != rval
                try:
                    if op == ">=":
                        return lval >= rval
                    if op == "<=":
                        return lval <= rval
                    if op == ">":
                        return lval > rval
                    if op == "<":
                        return lval < rval
                except TypeError:
                    return False
        # IN check: var.prop IN [...]
        m = re.match(
            r"^(?P<lhs>[\w.]+)\s+IN\s+\[(?P<rhs>.+)\]$",
            clause,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            lval = self._eval_return_expr(m.group("lhs"), row)
            # Only support $param-name inside IN
            inner = m.group("rhs").strip()
            if inner.startswith("$"):
                seq = parameters.get(inner[1:])
                return lval in (seq or [])
            # Fall back to literal list parse
            try:
                literal = eval(inner, {"__builtins__": {}}, {})  # noqa: S307
            except Exception:
                return False
            return lval in literal
        return True  # unknown clause: permissive

    def _eval_value(
        self, token: str, parameters: Dict[str, Any]
    ) -> Any:
        token = token.strip()
        # $param
        if token.startswith("$"):
            return parameters.get(token[1:])
        # Quoted literal
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            return token[1:-1]
        # Numeric
        try:
            if "." in token:
                return float(token)
            return int(token)
        except ValueError:
            pass
        # Bareword: treat as string literal
        return token

    def _eval_return_expr(self, expr: str, row: Dict[str, Any]) -> Any:
        # labels(var)
        m = re.match(r"^labels\s*\(\s*(?P<var>\w+)\s*\)$", expr, re.IGNORECASE)
        if m:
            v = m.group("var")
            node = row.get(v)
            if isinstance(node, dict):
                return list(node.get("labels") or [])
            return []
        # var.prop
        if "." in expr:
            var, _, prop = expr.partition(".")
            val = row.get(var)
            if isinstance(val, dict):
                inner = val
                # Support nested property access via dotted path
                for part in prop.split("."):
                    if isinstance(inner, dict):
                        inner = inner.get(part)
                    elif inner is None:
                        return None
                    else:
                        # properties nested access
                        return None
                return inner
            return None
        # Bare var
        return row.get(expr)

    def _apply_order_by(
        self, body: str, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        body = body.strip()
        desc = False
        if body.upper().endswith("DESC"):
            body = body[:-4].strip()
            desc = True
        elif body.upper().endswith("ASC"):
            body = body[:-3].strip()

        def sort_key(r: Dict[str, Any]) -> Any:
            val = self._eval_return_expr(body, r)
            # Always tuple-sort: (None_flag, value)
            if val is None:
                return (0, 0)
            return (1, val)

        try:
            return sorted(rows, key=sort_key, reverse=desc)
        except TypeError:
            return rows

    @staticmethod
    def _split_return(body: str) -> List[str]:
        # Split on commas not inside parentheses
        out: List[str] = []
        depth = 0
        cur: List[str] = []
        for ch in body:
            if ch == "(":
                depth += 1
                cur.append(ch)
            elif ch == ")":
                depth -= 1
                cur.append(ch)
            elif ch == "," and depth == 0:
                out.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
        return out

    @staticmethod
    def _node_to_row(node: _Node) -> Dict[str, Any]:
        return {
            "id": node.node_id,
            "node_id": node.node_id,
            "labels": list(node.labels),
            "properties": dict(node.properties),
            # Flatten properties for direct access like n.text
            **dict(node.properties),
        }

    def _drop_edge(self, edge: _Edge) -> None:
        out_list = self._out.get(edge.source_id, [])
        self._out[edge.source_id] = [e for e in out_list if e is not edge]
        in_list = self._in.get(edge.target_id, [])
        self._in[edge.target_id] = [e for e in in_list if e is not edge]

    # ------------------------------------------------------------------ #
    # Efficient override of _iter_nodes
    # ------------------------------------------------------------------ #
    def _iter_nodes(self) -> List[Tuple[str, List[str], Dict[str, Any]]]:
        return [
            (nid, list(node.labels), dict(node.properties))
            for nid, node in self._nodes.items()
        ]

    def count_nodes(self) -> int:
        return len(self._nodes)

    def count_edges(self) -> int:
        return sum(len(es) for es in self._out.values())


__all__ = ["InMemoryGraphStorage"]
