"""Tests for MemoryGraph — the most important file in the framework.

Covers topology construction, tree queries (SUBSUMES), the three core
algorithms (fan-out read, fan-in write, consolidation), per-edge policy
gating, cycle safety, and the forget sweep.
"""
from __future__ import annotations

import unittest

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryBuilder, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph.edge import EdgeKind, MemoryEdge
from unimem.graph.graph import MemoryGraph
from unimem.graph.node import MemoryNode
from unimem.policies.consolidation_policy import ConsolidationPolicy
from unimem.policies.forget_policy import ForgetPolicy
from unimem.policies.read_policy import ConcatRead, FirstNonEmptyRead
from unimem.policies.write_policy import AlwaysWrite, LambdaWritePolicy, NeverWrite


# --------------------------------------------------------------------------- #
# Test fixture: a recording stub module
# --------------------------------------------------------------------------- #
class RecordingMem(MemoryModule):
    """Stores every entry in insertion order; records calls for assertions."""

    def __init__(self, name: str = "rec"):
        self.name = name
        self._entries = []
        self.write_calls = []
        self.read_calls = []
        self.update_calls = 0
        self.consolidate_calls = 0
        self._consolidated_returns = []
        self.forget_calls = 0

    def write(self, entry, context):
        self.write_calls.append(entry.entry_id)
        self._entries.append(entry)
        return True

    def read(self, query):
        self.read_calls.append(query)
        # Return a copy so callers can mutate safely.
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()

    def stats(self):
        return {"count": len(self._entries), "name": self.name}

    def update(self, context):
        self.update_calls += 1

    def consolidate(self, other, context):
        self.consolidate_calls += 1
        out = list(self._consolidated_returns)
        self._consolidated_returns = []
        return out

    def set_consolidate_returns(self, entries):
        self._consolidated_returns = list(entries)

    def forget(self, n):  # used by a custom ForgetPolicy below
        n = min(n, len(self._entries))
        self._entries = self._entries[n:]
        return n


def _make_node(node_id: str, slot: MemorySlot, name: str = "rec") -> MemoryNode:
    return MemoryNode(node_id=node_id, slot=slot, module=RecordingMem(name))


# --------------------------------------------------------------------------- #
# Topology tests
# --------------------------------------------------------------------------- #
class TestTopology(unittest.TestCase):
    def test_add_and_get_node(self):
        g = MemoryGraph()
        n = _make_node("wm", MemorySlot.WM)
        g.add_node(n)
        self.assertIs(g.get_node("wm"), n)
        self.assertEqual(len(g), 1)
        self.assertIn("wm", g)

    def test_duplicate_node_id_raises(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        with self.assertRaises(ValueError):
            g.add_node(_make_node("wm", MemorySlot.WM))

    def test_add_edge_unknown_node_raises(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        with self.assertRaises(KeyError):
            g.add_edge(MemoryEdge("wm", "ghost", EdgeKind.FEEDS))

    def test_self_loop_edge_rejected(self):
        with self.assertRaises(ValueError):
            MemoryEdge("a", "a", EdgeKind.FEEDS)

    def test_edge_lookup_by_kind(self):
        g = MemoryGraph()
        for nid in ("wm", "em", "sm"):
            g.add_node(_make_node(nid, MemorySlot.WM))
        e1 = MemoryEdge("wm", "em", EdgeKind.FEEDS)
        e2 = MemoryEdge("em", "sm", EdgeKind.CONSOLIDATES_TO)
        g.add_edge(e1)
        g.add_edge(e2)
        self.assertEqual(g.edges_of("em", EdgeKind.FEEDS), [])
        self.assertEqual(g.edges_of("em", EdgeKind.CONSOLIDATES_TO), [e2])
        self.assertEqual(g.edges_into("em", EdgeKind.FEEDS), [e1])


# --------------------------------------------------------------------------- #
# Tree queries (SUBSUMES)
# --------------------------------------------------------------------------- #
class TestTreeQueries(unittest.TestCase):
    def _build_tree(self):
        # room -> area1, area2 ; area1 -> obj1, obj2 ; area2 -> obj3
        g = MemoryGraph()
        for nid, slot in [
            ("room", MemorySlot.SG),
            ("area1", MemorySlot.SG),
            ("area2", MemorySlot.SG),
            ("obj1", MemorySlot.SG),
            ("obj2", MemorySlot.SG),
            ("obj3", MemorySlot.SG),
        ]:
            g.add_node(_make_node(nid, slot))
        for src, tgt in [
            ("room", "area1"),
            ("room", "area2"),
            ("area1", "obj1"),
            ("area1", "obj2"),
            ("area2", "obj3"),
        ]:
            g.add_edge(MemoryEdge(src, tgt, EdgeKind.SUBSUMES))
        return g

    def test_get_children(self):
        g = self._build_tree()
        self.assertEqual(sorted(g.get_children("room")), ["area1", "area2"])
        self.assertEqual(sorted(g.get_children("area1")), ["obj1", "obj2"])
        self.assertEqual(g.get_children("obj1"), [])

    def test_get_parent(self):
        g = self._build_tree()
        self.assertEqual(g.get_parent("area1"), "room")
        self.assertEqual(g.get_parent("obj3"), "area2")
        self.assertIsNone(g.get_parent("room"))

    def test_get_subtree_bfs(self):
        g = self._build_tree()
        # room's subtree should include all 5 descendants, no duplicates
        sub = g.get_subtree("room")
        self.assertEqual(len(sub), 5)
        self.assertEqual(set(sub), {"area1", "area2", "obj1", "obj2", "obj3"})
        # BFS order: areas come before their objects
        self.assertLess(sub.index("area1"), sub.index("obj1"))
        self.assertLess(sub.index("area1"), sub.index("obj2"))
        self.assertLess(sub.index("area2"), sub.index("obj3"))

    def test_subtree_unknown_node_raises(self):
        g = self._build_tree()
        with self.assertRaises(KeyError):
            g.get_subtree("nonexistent")


# --------------------------------------------------------------------------- #
# Algorithm 1: fan-out read
# --------------------------------------------------------------------------- #
class TestFanOutRead(unittest.TestCase):
    def setUp(self):
        self.g = MemoryGraph()
        self.wm = _make_node("wm", MemorySlot.WM, "wm")
        self.em = _make_node("em", MemorySlot.EM, "em")
        self.sm = _make_node("sm", MemorySlot.SM, "sm")
        for n in (self.wm, self.em, self.sm):
            self.g.add_node(n)
        # Each holds a different entry
        self.e_wm = MemoryEntry("wm1", "current obs", semantic_keys=["now"])
        self.e_em = MemoryEntry("em1", "past event", semantic_keys=["past"])
        self.e_sm = MemoryEntry("sm1", "the chair is red", semantic_keys=["chair", "red"])
        self.wm.module.write(self.e_wm, MemoryContext())
        self.em.module.write(self.e_em, MemoryContext())
        self.sm.module.write(self.e_sm, MemoryContext())

    def test_no_slot_filter_reads_all(self):
        results = self.g.read(Query())
        self.assertEqual(len(results), 3)
        ids = {r.source_node_id for r in results}
        self.assertEqual(ids, {"wm", "em", "sm"})

    def test_slot_filter_restricts_targets(self):
        results = self.g.read(QueryBuilder().with_slot(MemorySlot.SM).build())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_node_id, "sm")
        self.assertEqual(results[0].entries[0].entry_id, "sm1")

    def test_slot_filter_accepts_multiple_slots(self):
        q = QueryBuilder().with_slot(MemorySlot.WM, MemorySlot.EM).build()
        results = self.g.read(q)
        self.assertEqual({r.source_node_id for r in results}, {"wm", "em"})

    def test_provenance_is_stamped(self):
        results = self.g.read(Query())
        for r in results:
            self.assertIsNotNone(r.source_node_id)
            self.assertIsNotNone(r.source_slot)

    def test_top_k_truncates_per_node(self):
        # Add more entries to wm
        self.wm.module.write(MemoryEntry("wm2", "x"), MemoryContext())
        self.wm.module.write(MemoryEntry("wm3", "y"), MemoryContext())
        results = self.g.read(QueryBuilder().with_slot(MemorySlot.WM).with_top_k(1).build())
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].entries), 1)

    def test_merge_results_uses_graph_default_policy(self):
        merged = self.g.merge_results(self.g.read(Query()))
        self.assertEqual(len(merged.entries), 3)
        self.assertEqual(merged.metadata["n_sources"], 3)

    def test_custom_read_policy(self):
        g = MemoryGraph(default_read_policy=FirstNonEmptyRead())
        wm = _make_node("wm", MemorySlot.WM)
        em = _make_node("em", MemorySlot.EM)
        g.add_node(wm)
        g.add_node(em)
        em.module.write(MemoryEntry("e", "x"), MemoryContext())
        merged = g.merge_results(g.read(Query()))
        self.assertEqual(len(merged.entries), 1)
        self.assertEqual(merged.source_node_id, "em")


# --------------------------------------------------------------------------- #
# Algorithm 2: fan-in write
# --------------------------------------------------------------------------- #
class TestFanInWrite(unittest.TestCase):
    def test_single_seed_no_feeds(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        entry = MemoryEntry("e1", "obs")
        result = g.write(entry, MemoryContext(), source_node_id="wm")
        self.assertEqual(result, {"wm": True})
        self.assertEqual(g.get_node("wm").module.write_calls, ["e1"])

    def test_propagates_along_feeds(self):
        g = MemoryGraph()
        wm = _make_node("wm", MemorySlot.WM)
        em = _make_node("em", MemorySlot.EM)
        sm = _make_node("sm", MemorySlot.SM)
        for n in (wm, em, sm):
            g.add_node(n)
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))
        g.add_edge(MemoryEdge("em", "sm", EdgeKind.FEEDS))

        entry = MemoryEntry("e1", "obs")
        result = g.write(entry, MemoryContext(), source_node_id="wm")
        self.assertEqual(result, {"wm": True, "em": True, "sm": True})
        for nid in ("wm", "em", "sm"):
            self.assertEqual(g.get_node(nid).module.write_calls, ["e1"])

    def test_no_source_writes_to_roots(self):
        # wm -> em ; sm is isolated (also a root).
        g = MemoryGraph()
        wm = _make_node("wm", MemorySlot.WM)
        em = _make_node("em", MemorySlot.EM)
        sm = _make_node("sm", MemorySlot.SM)
        for n in (wm, em, sm):
            g.add_node(n)
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))

        entry = MemoryEntry("e1", "obs")
        result = g.write(entry, MemoryContext())
        # Roots = wm, sm (em has incoming FEEDS so it's not a root)
        # Propagation reaches wm + em (via wm -> em) and sm (isolated).
        self.assertEqual(set(result.keys()), {"wm", "em", "sm"})
        self.assertTrue(all(result.values()))

    def test_no_source_no_roots_writes_all(self):
        # Cyclic FEEDS graph: every node has incoming FEEDS.
        g = MemoryGraph()
        a = _make_node("a", MemorySlot.WM)
        b = _make_node("b", MemorySlot.WM)
        g.add_node(a)
        g.add_node(b)
        g.add_edge(MemoryEdge("a", "b", EdgeKind.FEEDS))
        g.add_edge(MemoryEdge("b", "a", EdgeKind.FEEDS))
        result = g.write(MemoryEntry("e", "x"), MemoryContext())
        # Both written once; no infinite loop.
        self.assertEqual(set(result.keys()), {"a", "b"})

    def test_cycle_safe_via_visited(self):
        # a -> b -> c -> a
        g = MemoryGraph()
        for nid in ("a", "b", "c"):
            g.add_node(_make_node(nid, MemorySlot.WM))
        g.add_edge(MemoryEdge("a", "b", EdgeKind.FEEDS))
        g.add_edge(MemoryEdge("b", "c", EdgeKind.FEEDS))
        g.add_edge(MemoryEdge("c", "a", EdgeKind.FEEDS))
        result = g.write(MemoryEntry("e", "x"), MemoryContext(), source_node_id="a")
        self.assertEqual(set(result), {"a", "b", "c"})
        for nid in ("a", "b", "c"):
            self.assertEqual(len(g.get_node(nid).module.write_calls), 1)

    def test_edge_write_policy_gates_propagation(self):
        # wm -> em (gated by NeverWrite) -> sm
        g = MemoryGraph()
        for nid, slot in [("wm", MemorySlot.WM), ("em", MemorySlot.EM), ("sm", MemorySlot.SM)]:
            g.add_node(_make_node(nid, slot))
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS, policy=NeverWrite()))
        g.add_edge(MemoryEdge("em", "sm", EdgeKind.FEEDS))
        result = g.write(MemoryEntry("e", "x"), MemoryContext(), source_node_id="wm")
        # wm written; em gated off; sm never reached
        self.assertEqual(result, {"wm": True, "em": False})
        self.assertEqual(g.get_node("sm").module.write_calls, [])

    def test_module_write_policy_used_when_no_edge_policy(self):
        g = MemoryGraph()
        wm = _make_node("wm", MemorySlot.WM)
        em = _make_node("em", MemorySlot.EM)
        em.module.write_policy = NeverWrite()  # module-level policy
        g.add_node(wm)
        g.add_node(em)
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))
        result = g.write(MemoryEntry("e", "x"), MemoryContext(), source_node_id="wm")
        self.assertEqual(result, {"wm": True, "em": False})

    def test_default_write_policy_when_no_other_set(self):
        g = MemoryGraph(default_write_policy=NeverWrite())
        g.add_node(_make_node("wm", MemorySlot.WM))
        result = g.write(MemoryEntry("e", "x"), MemoryContext(), source_node_id="wm")
        self.assertEqual(result, {"wm": False})

    def test_event_triggered_write_via_lambda_policy(self):
        # Models INHerit-SG: only ingest when entry metadata says "topology_changed"
        g = MemoryGraph()
        wm = _make_node("wm", MemorySlot.WM)
        sg = _make_node("sg", MemorySlot.SG)
        g.add_node(wm)
        g.add_node(sg)
        trigger_pol = LambdaWritePolicy(
            lambda m, e, c: e.metadata.get("topology_changed", False)
        )
        g.add_edge(MemoryEdge("wm", "sg", EdgeKind.FEEDS, policy=trigger_pol))

        # Boring observation: not propagated
        boring = MemoryEntry("b", "same scene")
        r1 = g.write(boring, MemoryContext(), source_node_id="wm")
        self.assertEqual(r1, {"wm": True, "sg": False})

        # Topology-changing observation: propagated
        event = MemoryEntry("t", "new object!", metadata={"topology_changed": True})
        r2 = g.write(event, MemoryContext(), source_node_id="wm")
        self.assertEqual(r2["sg"], True)
        self.assertEqual(g.get_node("sg").module.write_calls, ["t"])

    def test_source_stamps_entry_provenance(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        entry = MemoryEntry("e", "x")
        g.write(entry, MemoryContext(), source_node_id="wm")
        self.assertEqual(entry.source_slot, MemorySlot.WM.value)

    def test_unknown_source_node_raises(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        with self.assertRaises(KeyError):
            g.write(MemoryEntry("e", "x"), MemoryContext(), source_node_id="ghost")

    def test_empty_graph_returns_empty(self):
        self.assertEqual(MemoryGraph().write(MemoryEntry("e", "x"), MemoryContext()), {})


# --------------------------------------------------------------------------- #
# Algorithm 3: consolidation pass
# --------------------------------------------------------------------------- #
class _StubFactPolicy(ConsolidationPolicy):
    """Extracts one canned fact regardless of source/target."""

    def __init__(self, fact_text="extracted fact"):
        self.fact_text = fact_text
        self.calls = []

    def extract(self, source, target, context):
        self.calls.append((source, target))
        return [MemoryEntry("fact-" + self.fact_text, self.fact_text, source_slot="em")]


class _CountForget(ForgetPolicy):
    """Trims the module down to at most ``cap`` entries (FIFO drop)."""

    def __init__(self, cap):
        self.cap = cap
        self.applied_to = []

    def apply(self, module, context):
        self.applied_to.append(module)
        if not hasattr(module, "forget"):
            return 0
        current = module.stats().get("count", 0)
        if current <= self.cap:
            return 0
        return module.forget(current - self.cap)


class TestConsolidation(unittest.TestCase):
    def test_edge_policy_extracted_and_stored(self):
        g = MemoryGraph()
        em = _make_node("em", MemorySlot.EM)
        sm = _make_node("sm", MemorySlot.SM)
        g.add_node(em)
        g.add_node(sm)
        pol = _StubFactPolicy("the chair is red")
        g.add_edge(MemoryEdge("em", "sm", EdgeKind.CONSOLIDATES_TO, policy=pol))

        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["extracted"]["em"], 1)
        self.assertEqual(result["stored"]["sm"], 1)
        self.assertEqual(len(sm.module.write_calls), 1)
        self.assertEqual(pol.calls, [(em.module, sm.module)])

    def test_no_edge_policy_falls_back_to_module_consolidate(self):
        g = MemoryGraph()
        em = _make_node("em", MemorySlot.EM)
        sm = _make_node("sm", MemorySlot.SM)
        em.module.set_consolidate_returns([MemoryEntry("c1", "from-module")])
        g.add_node(em)
        g.add_node(sm)
        # No policy on the edge → Passthrough uses source.consolidate()
        g.add_edge(MemoryEdge("em", "sm", EdgeKind.CONSOLIDATES_TO))
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["extracted"]["em"], 1)
        self.assertEqual(result["stored"]["sm"], 1)
        self.assertEqual(em.module.consolidate_calls, 1)

    def test_no_consolidates_to_edges_is_noop(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        g.add_node(_make_node("em", MemorySlot.EM))
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(sum(result["extracted"].values()), 0)
        self.assertEqual(sum(result["stored"].values()), 0)

    def test_one_source_consolidates_to_multiple_targets(self):
        # EM → SM (extract facts) ; EM → GM (extract spatial patterns)
        g = MemoryGraph()
        em = _make_node("em", MemorySlot.EM)
        sm = _make_node("sm", MemorySlot.SM)
        gm = _make_node("gm", MemorySlot.GM)
        for n in (em, sm, gm):
            g.add_node(n)
        fact_pol = _StubFactPolicy("fact")
        spatial_pol = _StubFactPolicy("spatial")
        g.add_edge(MemoryEdge("em", "sm", EdgeKind.CONSOLIDATES_TO, policy=fact_pol))
        g.add_edge(MemoryEdge("em", "gm", EdgeKind.CONSOLIDATES_TO, policy=spatial_pol))
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["stored"]["sm"], 1)
        self.assertEqual(result["stored"]["gm"], 1)
        # Same source was extracted from twice
        self.assertEqual(result["extracted"]["em"], 2)

    def test_forget_sweep_runs_after_consolidation(self):
        g = MemoryGraph()
        em = _make_node("em", MemorySlot.EM)
        for i in range(5):
            em.module.write(MemoryEntry(f"e{i}", "x"), MemoryContext())
        g.add_node(em)
        g.default_forget_policy = _CountForget(cap=2)
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["forgotten"]["em"], 3)
        self.assertEqual(em.module.stats()["count"], 2)

    def test_module_forget_policy_overrides_graph_default(self):
        g = MemoryGraph()
        em = _make_node("em", MemorySlot.EM)
        # Graph says "forget everything"; module says "forget nothing"
        g.default_forget_policy = _CountForget(cap=0)
        em.module.forget_policy = type(  # quick NoOp-like
            "M", (ForgetPolicy,), {"apply": lambda self, m, c: 0}
        )()
        for i in range(3):
            em.module.write(MemoryEntry(f"e{i}", "x"), MemoryContext())
        g.add_node(em)
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["forgotten"]["em"], 0)
        self.assertEqual(em.module.stats()["count"], 3)


# --------------------------------------------------------------------------- #
# update_all + summary
# --------------------------------------------------------------------------- #
class TestUpdateAndSummary(unittest.TestCase):
    def test_update_all_calls_every_module(self):
        g = MemoryGraph()
        wm = _make_node("wm", MemorySlot.WM)
        em = _make_node("em", MemorySlot.EM)
        for n in (wm, em):
            g.add_node(n)
        g.update_all(MemoryContext())
        self.assertEqual(wm.module.update_calls, 1)
        self.assertEqual(em.module.update_calls, 1)

    def test_summary_structure(self):
        g = MemoryGraph()
        g.add_node(_make_node("wm", MemorySlot.WM))
        g.add_node(_make_node("em", MemorySlot.EM))
        g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))
        s = g.summary()
        self.assertEqual(s["n_nodes"], 2)
        self.assertEqual(s["n_edges"], 1)
        self.assertEqual(s["edges_by_kind"]["FEEDS"], 1)
        self.assertIn("wm", s["nodes"])
        self.assertEqual(s["nodes"]["wm"]["slot"], "WM")


if __name__ == "__main__":
    unittest.main()
