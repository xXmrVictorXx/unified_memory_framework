"""Tests for CLiViS → unimem graph wiring."""
from __future__ import annotations

import unittest

from reproductions.clivis.graph_spec import build_clivis_graph
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.graph.edge import EdgeKind


class TestCLiViSGraph(unittest.TestCase):
    def test_build_graph(self):
        g = build_clivis_graph(question="q")
        self.assertEqual(len(g), 3)
        kinds = sorted(e.kind.name for e in g.edges())
        self.assertEqual(kinds, ["FEEDS", "FEEDS", "INDEXES"])

    def test_wm_propagates_to_sg_and_gm(self):
        g = build_clivis_graph(question="q")
        # First add a period to GM so the area write makes sense
        g.get_node("gm").module._init_period("00:00:00-00:00:30", "")
        # WM-style entry that FEEDS will propagate
        entry = MemoryEntry(
            "obs1", "saw a cup in kitchen",
            metadata={
                "kind": "object",
                "name": "cup",
                "period": "00:00:00-00:00:30",
            },
        )
        result = g.write(entry, MemoryContext(), source_node_id="wm")
        self.assertTrue(result["wm"])
        self.assertTrue(result["sg"])
        self.assertTrue(result["gm"])
        # GM has the cup registered
        self.assertIn("cup", g.get_node("gm").module.periods_to_obj_names["00:00:00-00:00:30"])

    def test_summary_runs(self):
        g = build_clivis_graph()
        s = g.summary()
        self.assertEqual(s["n_nodes"], 3)
        for nid in ("wm", "sg", "gm"):
            self.assertIn(nid, s["nodes"])


if __name__ == "__main__":
    unittest.main()
