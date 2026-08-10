"""Tests for the R4 → unimem graph wiring."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockEmbedding
from reproductions.r4.graph_spec import build_r4_graph
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.graph.edge import EdgeKind


class TestR4Graph(unittest.TestCase):
    def test_build_graph(self):
        g = build_r4_graph(embedding_fn=MockEmbedding(dim=32))
        self.assertEqual(len(g), 3)
        kinds = sorted(e.kind.name for e in g.edges())
        self.assertEqual(kinds, ["CONSOLIDATES_TO", "FEEDS"])

    def test_observation_propagates_wm_to_db(self):
        g = build_r4_graph(embedding_fn=MockEmbedding(dim=32))
        obs = MemoryEntry(
            entry_id="obs1",
            text="a red chair",
            spatial_keys=[(1.0, 2.0, 0.5, 0.5, 0.5, 1.0)],
            temporal_keys=[10.0],
            source_slot="working_memory",
        )
        result = g.write(obs, MemoryContext(timestamp=10.0), source_node_id="wm")
        self.assertTrue(result["wm"])
        self.assertTrue(result["db"])

    def test_db_node_is_r4_instance(self):
        from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase

        g = build_r4_graph(embedding_fn=MockEmbedding(dim=32))
        db_node = g.get_node("db")
        self.assertIsInstance(db_node.module, R4KnowledgeDatabase)

    def test_summary_runs(self):
        g = build_r4_graph(embedding_fn=MockEmbedding(dim=32))
        s = g.summary()
        self.assertEqual(s["n_nodes"], 3)
        self.assertIn("db", s["nodes"])


if __name__ == "__main__":
    unittest.main()
