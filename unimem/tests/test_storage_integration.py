"""Tests for the storage-aware MemoryGraph + MemoryModule integration."""
from __future__ import annotations

import unittest

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import QueryBuilder
from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind, MemoryEdge
from unimem.graph.graph import MemoryGraph
from unimem.graph.node import MemoryNode
from unimem.graph_storage import InMemoryGraphStorage
from unimem.op_log import OpLogEntry, SQLiteOpLog


def _build_graph():
    gs = InMemoryGraphStorage()
    wm = MemoryModule(slot=MemorySlot.WM, graph_storage=gs)
    em = MemoryModule(slot=MemorySlot.EM, graph_storage=gs)
    graph = MemoryGraph(graph_storage=gs)
    graph.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=wm))
    graph.add_node(MemoryNode(node_id="em", slot=MemorySlot.EM, module=em))
    graph.add_edge(MemoryEdge(source_id="wm", target_id="em", kind=EdgeKind.FEEDS))
    return graph, gs


class TestGraphPersistsTopology(unittest.TestCase):
    def test_add_node_mirrors_to_storage(self):
        graph, gs = _build_graph()
        node = gs.get_node("wm")
        self.assertIsNotNone(node)
        self.assertIn("ModuleNode", node["labels"])

    def test_add_edge_mirrors_to_storage(self):
        graph, gs = _build_graph()
        edges = gs.query("MATCH (a)-[r:feeds]->(b) RETURN a, b")
        self.assertEqual(len(edges), 1)


class TestGraphWritePersists(unittest.TestCase):
    def test_write_creates_storage_nodes(self):
        graph, gs = _build_graph()
        entry = MemoryEntry(
            entry_id="e1",
            text="person walking",
            semantic_keys=["person"],
        )
        ctx = MemoryContext(timestamp=1.0)
        results = graph.write(entry, ctx, source_node_id="wm")
        # Both wm and em should be True (FEEDS propagation)
        self.assertTrue(results.get("wm"))
        self.assertTrue(results.get("em"))
        # Storage should have an episodic node with that id
        em_node = gs.get_node("e1")
        self.assertIsNotNone(em_node)
        self.assertIn(MemorySlot.EM.value, em_node["labels"])

    def test_read_returns_storage_entries(self):
        graph, gs = _build_graph()
        entry = MemoryEntry(
            entry_id="e1", text="hello", semantic_keys=["greeting"]
        )
        graph.write(entry, MemoryContext(), source_node_id="wm")
        results = graph.read(QueryBuilder().with_slot(MemorySlot.EM).build())
        self.assertGreater(len(results), 0)
        # The EM module's read should find the entry
        em_result = [r for r in results if r.source_slot == "episodic"]
        self.assertEqual(len(em_result), 1)
        self.assertEqual(em_result[0].entries[0].entry_id, "e1")


class TestGraphOpLog(unittest.TestCase):
    def test_op_log_records_writes(self):
        gs = InMemoryGraphStorage()
        with SQLiteOpLog(":memory:") as log:
            graph = MemoryGraph(graph_storage=gs, op_log=log)
            wm = MemoryModule(slot=MemorySlot.WM, graph_storage=gs)
            graph.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=wm))
            graph.write(
                MemoryEntry(entry_id="e1", text="hello"),
                MemoryContext(),
                source_node_id="wm",
            )
            # Should have one write + one write_done entry
            writes = log.query(op_type="write")
            self.assertEqual(len(writes), 1)
            write_dones = log.query(op_type="write_done")
            self.assertEqual(len(write_dones), 1)
            self.assertEqual(writes[0].node_id, "wm")
            self.assertEqual(writes[0].entry_dict["entry_id"], "e1")

    def test_op_log_replay_is_idempotent(self):
        """Replaying the log against a fresh graph + storage re-creates state."""
        gs1 = InMemoryGraphStorage()
        with SQLiteOpLog(":memory:") as log:
            graph1 = MemoryGraph(graph_storage=gs1, op_log=log)
            wm = MemoryModule(slot=MemorySlot.WM, graph_storage=gs1)
            em = MemoryModule(slot=MemorySlot.EM, graph_storage=gs1)
            graph1.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=wm))
            graph1.add_node(MemoryNode(node_id="em", slot=MemorySlot.EM, module=em))
            graph1.add_edge(MemoryEdge(source_id="wm", target_id="em", kind=EdgeKind.FEEDS))
            graph1.write(
                MemoryEntry(entry_id="e1", text="hello"),
                MemoryContext(),
                source_node_id="wm",
            )

            # Simulate crash: spin up a fresh storage and replay
            gs2 = InMemoryGraphStorage()
            graph2 = MemoryGraph(graph_storage=gs2, op_log=log)
            wm2 = MemoryModule(slot=MemorySlot.WM, graph_storage=gs2)
            em2 = MemoryModule(slot=MemorySlot.EM, graph_storage=gs2)
            graph2.add_node(MemoryNode(node_id="wm", slot=MemorySlot.WM, module=wm2))
            graph2.add_node(MemoryNode(node_id="em", slot=MemorySlot.EM, module=em2))
            graph2.add_edge(MemoryEdge(source_id="wm", target_id="em", kind=EdgeKind.FEEDS))
            for entry in log.replay():
                if entry.op_type == "write" and entry.entry_dict:
                    graph2.write(
                        MemoryEntry(
                            entry_id=entry.entry_dict["entry_id"],
                            text=entry.entry_dict["text"],
                        ),
                        MemoryContext(),
                        source_node_id=entry.node_id,
                    )
            # The replayed state should contain the original entry
            node = gs2.get_node("e1")
            self.assertIsNotNone(node)
            self.assertIn(MemorySlot.EM.value, node["labels"])


if __name__ == "__main__":
    unittest.main()
