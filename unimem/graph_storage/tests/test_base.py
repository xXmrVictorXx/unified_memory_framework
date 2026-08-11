"""Tests for the GraphStorage ABC + InMemoryGraphStorage backend."""
from __future__ import annotations

import unittest

from unimem.core.entry import MemoryEntry
from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind
from unimem.graph_storage import (
    InMemoryGraphStorage,
    create_graph_storage,
)
from unimem.graph_storage.base import GraphStorage


class TestGraphStorageFactory(unittest.TestCase):
    def test_default_is_in_memory(self):
        gs = create_graph_storage({})
        self.assertIsInstance(gs, InMemoryGraphStorage)

    def test_explicit_memory_backend(self):
        gs = create_graph_storage({"backend": "memory"})
        self.assertIsInstance(gs, InMemoryGraphStorage)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_graph_storage({"backend": "postgres"})


class TestInMemoryCRUD(unittest.TestCase):
    def setUp(self):
        self.gs = InMemoryGraphStorage()

    def test_add_and_get_node(self):
        ok = self.gs.add_node("n1", ["em"], {"text": "hello"})
        self.assertTrue(ok)
        node = self.gs.get_node("n1")
        self.assertIsNotNone(node)
        self.assertEqual(node["labels"], ["em"])
        self.assertEqual(node["properties"]["text"], "hello")

    def test_add_node_upsert_merges_labels(self):
        self.gs.add_node("n1", ["em"], {"text": "a"})
        self.gs.add_node("n1", ["time_index"], {"extra": 1})
        node = self.gs.get_node("n1")
        self.assertEqual(set(node["labels"]), {"em", "time_index"})
        self.assertEqual(node["properties"]["text"], "a")
        self.assertEqual(node["properties"]["extra"], 1)

    def test_get_missing_node_returns_none(self):
        self.assertIsNone(self.gs.get_node("nope"))

    def test_add_edge(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.assertTrue(self.gs.add_edge("a", "b", "FEEDS", {"policy": "always"}))
        neighbours = self.gs.get_neighbors("a", direction="out")
        self.assertEqual(len(neighbours), 1)
        self.assertEqual(neighbours[0][0], "b")
        self.assertEqual(neighbours[0][1], "FEEDS")

    def test_add_edge_missing_node_returns_false(self):
        self.assertFalse(self.gs.add_edge("a", "b", "FEEDS"))

    def test_edge_upsert_merges_properties(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS", {"k1": 1})
        self.gs.add_edge("a", "b", "FEEDS", {"k2": 2})
        neighbours = self.gs.get_neighbors("a", "FEEDS", "out")
        self.assertEqual(len(neighbours), 1)
        props = neighbours[0][2]
        self.assertEqual(props["k1"], 1)
        self.assertEqual(props["k2"], 2)

    def test_get_neighbors_in_direction(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        in_to_b = self.gs.get_neighbors("b", direction="in")
        self.assertEqual(len(in_to_b), 1)
        self.assertEqual(in_to_b[0][0], "a")

    def test_get_neighbors_both_directions(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "a", "INDEXES")
        both = self.gs.get_neighbors("a", direction="both")
        self.assertEqual(len(both), 2)
        types = {t for (_, t, _) in both}
        self.assertEqual(types, {"FEEDS", "INDEXES"})

    def test_delete_node(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.assertEqual(self.gs.delete_node("a"), 1)
        self.assertIsNone(self.gs.get_node("a"))
        # Edge should be gone from b's incoming
        self.assertEqual(self.gs.get_neighbors("b", direction="in"), [])

    def test_delete_edge(self):
        self.gs.add_node("a", ["x"], {})
        self.gs.add_node("b", ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("a", "b", "INDEXES")
        # Delete only FEEDS
        count = self.gs.delete_edge("a", "b", "FEEDS")
        self.assertEqual(count, 1)
        remaining = self.gs.get_neighbors("a", direction="out")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0][1], "INDEXES")

    def test_bfs(self):
        # Build a -> b -> c -> d chain plus a -> c shortcut
        for nid in ("a", "b", "c", "d"):
            self.gs.add_node(nid, ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "c", "FEEDS")
        self.gs.add_edge("c", "d", "FEEDS")
        self.gs.add_edge("a", "c", "INDEXES")
        # BFS along FEEDS only
        out = self.gs.bfs("a", rel_types=["FEEDS"])
        self.assertEqual(set(out), {"b", "c", "d"})
        # BFS along INDEXES only
        out = self.gs.bfs("a", rel_types=["INDEXES"])
        self.assertEqual(out, ["c"])
        # BFS along everything
        out = self.gs.bfs("a")
        self.assertEqual(set(out), {"b", "c", "d"})

    def test_bfs_respects_max_depth(self):
        for nid in ("a", "b", "c", "d"):
            self.gs.add_node(nid, ["x"], {})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "c", "FEEDS")
        self.gs.add_edge("c", "d", "FEEDS")
        self.assertEqual(set(self.gs.bfs("a", ["FEEDS"], max_depth=1)), {"b"})
        self.assertEqual(set(self.gs.bfs("a", ["FEEDS"], max_depth=2)), {"b", "c"})


class TestInMemoryQuery(unittest.TestCase):
    def setUp(self):
        self.gs = InMemoryGraphStorage()
        self.gs.add_node("a", ["em"], {"text": "alpha", "weight": 1})
        self.gs.add_node("b", ["em"], {"text": "beta", "weight": 2})
        self.gs.add_node("c", ["sm"], {"text": "gamma", "weight": 3})
        self.gs.add_edge("a", "b", "FEEDS")
        self.gs.add_edge("b", "c", "CONSOLIDATES_TO")

    def test_match_all_nodes(self):
        rows = self.gs.query("MATCH (n) RETURN n")
        self.assertEqual(len(rows), 3)

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

    def test_match_with_edge(self):
        rows = self.gs.query(
            "MATCH (a)-[r:FEEDS]->(b) RETURN a, b"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["a"]["node_id"], "a")
        self.assertEqual(rows[0]["b"]["node_id"], "b")

    def test_match_with_param(self):
        rows = self.gs.query(
            "MATCH (n:em) WHERE n.text = $t RETURN n",
            {"t": "beta"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"]["node_id"], "b")

    def test_count(self):
        rows = self.gs.query("MATCH (n:em) RETURN COUNT(*)")
        self.assertEqual(rows[0]["count"], 2)

    def test_detach_delete(self):
        self.gs.query("MATCH (n:em) DETACH DELETE n")
        rows = self.gs.query("MATCH (n) RETURN n")
        # Only the sm node should remain
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"]["node_id"], "c")


class TestMemoryLevelHelpers(unittest.TestCase):
    def setUp(self):
        self.gs = InMemoryGraphStorage()

    def _entry(self, eid, text, **kw):
        return MemoryEntry(entry_id=eid, text=text, **kw)

    def test_add_memory_node(self):
        e = self._entry("e1", "red chair", semantic_keys=["chair", "red"])
        ok = self.gs.add_memory_node(MemorySlot.EM, e)
        self.assertTrue(ok)
        node = self.gs.get_node("e1")
        self.assertIn(MemorySlot.EM.value, node["labels"])
        self.assertEqual(node["properties"]["text"], "red chair")
        self.assertEqual(
            set(node["properties"]["semantic_keys"]),
            {"chair", "red"},
        )

    def test_add_memory_node_extra_label(self):
        e = self._entry("e1", "Alice")
        self.gs.add_memory_node(
            MemorySlot.SG, e, extra_labels=["Person", "clivis"]
        )
        node = self.gs.get_node("e1")
        for lab in (MemorySlot.SG.value, "Person", "clivis"):
            self.assertIn(lab, node["labels"])

    def test_add_time_index_creates_timeindex_node(self):
        e = self._entry("e1", "event")
        self.gs.add_memory_node(MemorySlot.EM, e)
        ok = self.gs.add_time_index("e1", timestamp=1.5, clip_index=0)
        self.assertTrue(ok)
        # Find the TimeIndex neighbour
        neighbours = self.gs.get_neighbors("e1", "AT_TIME", "out")
        self.assertEqual(len(neighbours), 1)
        ti_id = neighbours[0][0]
        ti_node = self.gs.get_node(ti_id)
        self.assertIn("TimeIndex", ti_node["labels"])
        self.assertEqual(ti_node["properties"]["timestamp"], 1.5)
        self.assertEqual(ti_node["properties"]["clip_index"], 0)

    def test_query_memories_by_slot(self):
        self.gs.add_memory_node(
            MemorySlot.EM,
            self._entry("e1", "first", semantic_keys=["alpha"]),
        )
        self.gs.add_memory_node(
            MemorySlot.SG,
            self._entry("e2", "second", semantic_keys=["alpha"]),
        )
        em_only = self.gs.query_memories(slot=MemorySlot.EM)
        self.assertEqual([e.entry_id for e in em_only], ["e1"])
        sg_only = self.gs.query_memories(slot=MemorySlot.SG)
        self.assertEqual([e.entry_id for e in sg_only], ["e2"])

    def test_query_memories_by_semantic_keys(self):
        self.gs.add_memory_node(
            MemorySlot.EM,
            self._entry("e1", "first", semantic_keys=["alpha", "beta"]),
        )
        self.gs.add_memory_node(
            MemorySlot.EM,
            self._entry("e2", "second", semantic_keys=["gamma"]),
        )
        result = self.gs.query_memories(
            slot=MemorySlot.EM, semantic_keys=["beta"]
        )
        self.assertEqual([e.entry_id for e in result], ["e1"])

    def test_query_memories_with_time_index(self):
        e1 = self._entry("e1", "first")
        e2 = self._entry("e2", "second")
        self.gs.add_memory_node(MemorySlot.EM, e1)
        self.gs.add_memory_node(MemorySlot.EM, e2)
        self.gs.add_time_index("e1", timestamp=5.0)
        self.gs.add_time_index("e2", timestamp=20.0)
        # Time range filter
        result = self.gs.query_memories(
            slot=MemorySlot.EM, time_range=(0.0, 10.0)
        )
        ids = sorted(e.entry_id for e in result)
        self.assertEqual(ids, ["e1"])
        result = self.gs.query_memories(slot=MemorySlot.EM, time_range=(0.0, 30.0))
        self.assertEqual(sorted(e.entry_id for e in result), ["e1", "e2"])

    def test_query_memories_clip_index(self):
        e1 = self._entry("e1", "first")
        e2 = self._entry("e2", "second")
        self.gs.add_memory_node(MemorySlot.EM, e1)
        self.gs.add_memory_node(MemorySlot.EM, e2)
        self.gs.add_time_index("e1", clip_index=0)
        self.gs.add_time_index("e2", clip_index=1)
        result = self.gs.query_memories(slot=MemorySlot.EM, clip_index=1)
        self.assertEqual([e.entry_id for e in result], ["e2"])

    def test_get_time_indexed_nodes(self):
        e1 = self._entry("e1", "first")
        self.gs.add_memory_node(MemorySlot.EM, e1)
        self.gs.add_time_index("e1", clip_index=3, start_t=10.0, end_t=15.0)
        # Filter by clip_index
        result = self.gs.get_time_indexed_nodes(clip_index=3)
        self.assertEqual([e.entry_id for e in result], ["e1"])
        # Filter by time_range that overlaps
        result = self.gs.get_time_indexed_nodes(time_range=(12.0, 20.0))
        self.assertEqual([e.entry_id for e in result], ["e1"])
        # Filter by time_range that does NOT overlap
        result = self.gs.get_time_indexed_nodes(time_range=(20.0, 30.0))
        self.assertEqual(result, [])

    def test_add_module_node(self):
        ok = self.gs.add_module_node("wm", MemorySlot.WM, "default", label="wm")
        self.assertTrue(ok)
        node = self.gs.get_node("wm")
        self.assertIn("ModuleNode", node["labels"])
        self.assertIn(MemorySlot.WM.value, node["labels"])
        self.assertEqual(node["properties"]["impl"], "default")

    def test_add_module_edge(self):
        self.gs.add_module_node("wm", MemorySlot.WM, "default")
        self.gs.add_module_node("em", MemorySlot.EM, "default")
        ok = self.gs.add_module_edge(
            "wm", "em", EdgeKind.FEEDS, {"policy": "always"}
        )
        self.assertTrue(ok)
        neighbours = self.gs.get_neighbors("wm", "feeds", "out")
        self.assertEqual(len(neighbours), 1)
        self.assertEqual(neighbours[0][0], "em")


class TestGraphStorageABC(unittest.TestCase):
    def test_cannot_instantiate_abc_directly(self):
        with self.assertRaises(TypeError):
            GraphStorage()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
