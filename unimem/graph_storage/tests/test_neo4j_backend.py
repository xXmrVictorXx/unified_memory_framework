"""Smoke tests for Neo4jGraphStorage — require a running Neo4j instance.

Skipped automatically if Neo4j isn't reachable at bolt://localhost:7687.
Run the rest of the GraphStorage suite against Neo4j by setting
UNIMEMNeo4j_TEST=1; otherwise these tests are skipped.

To enable: ``docker-compose up -d`` then ``python -m unittest discover``.
"""
from __future__ import annotations

import unittest

from unimem.core.entry import MemoryEntry
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage
from unimem.graph_storage.base import GraphStorage


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return False
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password")
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


NEO4J_UP = _neo4j_available()
SKIP_REASON = "Neo4j not running at bolt://localhost:7687 (start with: docker-compose up -d)"


def _make_storage() -> GraphStorage:
    """Fresh Neo4j storage, cleared between tests."""
    from unimem.graph_storage.neo4j_backend import Neo4jGraphStorage
    return Neo4jGraphStorage(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        clear_on_init=True,
    )


@unittest.skipUnless(NEO4J_UP, SKIP_REASON)
class TestNeo4jCRUD(unittest.TestCase):
    def setUp(self):
        self.gs = _make_storage()

    def tearDown(self):
        # Leave the store clean for the next test.
        self.gs.query("MATCH (n) DETACH DELETE n")
        self.gs.close()

    def test_add_and_get_node(self):
        self.gs.add_node("n1", ["em"], {"text": "hello"})
        node = self.gs.get_node("n1")
        self.assertIsNotNone(node)
        self.assertIn("em", node["labels"])
        self.assertEqual(node["properties"]["text"], "hello")

    def test_node_id_is_unique_property(self):
        self.gs.add_node("n1", ["em"], {"text": "v1"})
        # MERGE on same id updates rather than creating a duplicate
        self.gs.add_node("n1", ["time_index"], {"extra": 1})
        node = self.gs.get_node("n1")
        self.assertIn("em", node["labels"])
        self.assertIn("time_index", node["labels"])
        self.assertEqual(node["properties"]["text"], "v1")
        self.assertEqual(node["properties"]["extra"], 1)

    def test_get_missing_node(self):
        self.assertIsNone(self.gs.get_node("nope"))

    def test_add_edge(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.assertTrue(self.gs.add_edge("a", "b", "FEEDS", {"policy": "always"}))
        nb = self.gs.get_neighbors("a", direction="out")
        self.assertEqual(len(nb), 1)
        self.assertEqual(nb[0][0], "b")
        self.assertEqual(nb[0][1], "FEEDS")
        self.assertEqual(nb[0][2]["policy"], "always")

    def test_edge_missing_endpoints_returns_false(self):
        self.assertFalse(self.gs.add_edge("ghost", "b", "FEEDS"))

    def test_get_neighbors_direction(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.assertEqual(len(self.gs.get_neighbors("a", direction="out")), 1)
        self.assertEqual(len(self.gs.get_neighbors("a", direction="in")), 0)
        self.assertEqual(len(self.gs.get_neighbors("b", direction="in")), 1)
        self.assertEqual(len(self.gs.get_neighbors("a", direction="both")), 1)

    def test_delete_node(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.assertEqual(self.gs.delete_node("a"), 1)
        self.assertIsNone(self.gs.get_node("a"))
        # Edges go with DETACH DELETE
        self.assertEqual(self.gs.get_neighbors("b", direction="in"), [])

    def test_delete_edge(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("a", "b", "INDEXES")
        self.assertEqual(self.gs.delete_edge("a", "b", "FEEDS"), 1)
        remaining = self.gs.get_neighbors("a", direction="out")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0][1], "INDEXES")

    def test_bfs(self):
        for nid in ("a", "b", "c", "d"):
            self.gs.add_node(nid, ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "c", "FEEDS")
        self.gs.add_edge("c", "d", "FEEDS")
        self.gs.add_edge("a", "c", "INDEXES")
        self.assertEqual(set(self.gs.bfs("a", ["FEEDS"])), {"b", "c", "d"})
        self.assertEqual(self.gs.bfs("a", ["INDEXES"]), ["c"])


@unittest.skipUnless(NEO4J_UP, SKIP_REASON)
class TestNeo4jQuery(unittest.TestCase):
    def setUp(self):
        self.gs = _make_storage()
        self.gs.add_node("a", ["em"], {"text": "alpha", "weight": 1})
        self.gs.add_node("b", ["em"], {"text": "beta", "weight": 2})
        self.gs.add_node("c", ["sm"], {"text": "gamma", "weight": 3})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "c", "CONSOLIDATES_TO")

    def tearDown(self):
        self.gs.query("MATCH (n) DETACH DELETE n")
        self.gs.close()

    def test_match_by_label(self):
        rows = self.gs.query("MATCH (n:em) RETURN n")
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("em", r["n"]["labels"])

    def test_match_with_where(self):
        rows = self.gs.query(
            "MATCH (n:em) WHERE n.weight >= 2 RETURN n"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"]["node_id"], "b")

    def test_count(self):
        rows = self.gs.query("MATCH (n:em) RETURN count(*) AS cnt")
        self.assertEqual(rows[0]["cnt"], 2)

    def test_match_with_edge(self):
        rows = self.gs.query(
            "MATCH (a:em)-[r:FEEDS]->(b) RETURN a, b"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["a"]["node_id"], "a")
        self.assertEqual(rows[0]["b"]["node_id"], "b")


@unittest.skipUnless(NEO4J_UP, SKIP_REASON)
class TestNeo4jMemoryHelpers(unittest.TestCase):
    """Memory-level convenience methods should work uniformly across backends."""

    def setUp(self):
        self.gs = _make_storage()

    def tearDown(self):
        self.gs.query("MATCH (n) DETACH DELETE n")
        self.gs.close()

    def test_add_memory_node_and_query(self):
        entry = MemoryEntry(
            entry_id="e1", text="red chair", semantic_keys=["chair", "red"]
        )
        self.gs.add_memory_node(MemorySlot.EM, entry)
        results = self.gs.query_memories(slot=MemorySlot.EM)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entry_id, "e1")
        self.assertIn("chair", results[0].semantic_keys)

    def test_time_index_round_trip(self):
        entry = MemoryEntry(entry_id="e1", text="event")
        self.gs.add_memory_node(MemorySlot.EM, entry)
        self.gs.add_time_index("e1", timestamp=5.0, clip_index=0)
        # Query by time range
        results = self.gs.query_memories(
            slot=MemorySlot.EM, time_range=(0.0, 10.0)
        )
        self.assertEqual([e.entry_id for e in results], ["e1"])
        # Query by clip_index
        results = self.gs.get_time_indexed_nodes(clip_index=0)
        self.assertEqual([e.entry_id for e in results], ["e1"])


@unittest.skipUnless(NEO4J_UP, SKIP_REASON)
class TestNeo4jInMemoryParity(unittest.TestCase):
    """Spot-check that Neo4j and InMemory backends give identical results."""

    def setUp(self):
        self.mem = InMemoryGraphStorage()
        self.neo = _make_storage()

    def tearDown(self):
        self.neo.query("MATCH (n) DETACH DELETE n")
        self.neo.close()

    def _seed(self, gs):
        gs.add_node("alice", ["scene_graph", "Person"], {"info": "host"})
        gs.add_node("kitchen", ["scene_graph", "Area"], {"time_range": "0-30"})
        gs.add_edge("alice", "kitchen", "LOCATED_IN", {"at": "start"})

    def test_node_lookup_identical(self):
        self._seed(self.mem)
        self._seed(self.neo)
        m = self.mem.get_node("alice")
        n = self.neo.get_node("alice")
        self.assertEqual(set(m["labels"]), set(n["labels"]))
        self.assertEqual(m["properties"]["info"], n["properties"]["info"])

    def test_neighbors_identical(self):
        self._seed(self.mem)
        self._seed(self.neo)
        m = self.mem.get_neighbors("alice", direction="out")
        n = self.neo.get_neighbors("alice", direction="out")
        self.assertEqual(len(m), len(n))
        self.assertEqual(
            sorted((t[0], t[1]) for t in m),
            sorted((t[0], t[1]) for t in n),
        )


if __name__ == "__main__":
    unittest.main()
