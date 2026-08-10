"""Tests for VideoHV-Agent → unimem graph wiring."""
from __future__ import annotations

import unittest

from reproductions.videohv.graph_spec import build_videohv_graph
from reproductions.videohv.memory.video_summary_memory import VideoSummaryMemory
from unimem.graph.edge import EdgeKind


class TestVideoHVGraph(unittest.TestCase):
    def test_build_graph(self):
        g = build_videohv_graph()
        self.assertEqual(len(g), 2)
        kinds = [e.kind.name for e in g.edges()]
        self.assertEqual(kinds, ["REFERENCES"])

    def test_summary_node_present(self):
        g = build_videohv_graph()
        self.assertIn("summary", g)
        self.assertIn("trace", g)

    def test_summary_memory_carried_through(self):
        mem = VideoSummaryMemory(action_captions=["a", "b"])
        g = build_videohv_graph(summary_memory=mem)
        self.assertIs(g.get_node("summary").module, mem)
        self.assertEqual(g.get_node("summary").module.stats()["count"], 2)

    def test_summary_runs(self):
        g = build_videohv_graph()
        s = g.summary()
        self.assertEqual(s["n_nodes"], 2)


if __name__ == "__main__":
    unittest.main()
