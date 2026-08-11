"""End-to-end tests for the reference ListEpisodicMemory + reference policies."""
from __future__ import annotations

import unittest

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryBuilder
from unimem.core.query import QueryResult
from unimem.core.slot_abc import EpisodicMemoryABC, SemanticMemoryABC
from unimem.core.slots import MemorySlot
from unimem.reference.consolidate_extract import ExtractFactsConsolidationPolicy
from unimem.reference.episodic_memory import ListEpisodicMemory
from unimem.reference.forget_fifo import FIFOForgetPolicy


class TestListEpisodicMemoryBasic(unittest.TestCase):
    def test_inherits_correct_abc(self):
        m = ListEpisodicMemory()
        self.assertIsInstance(m, EpisodicMemoryABC)
        self.assertIsInstance(m, MemoryModule)
        # Has default timescales
        self.assertEqual(m.timescales, (30.0, 180.0, 600.0, 3600.0))

    def test_custom_timescales(self):
        m = ListEpisodicMemory(timescales=(60.0,))
        self.assertEqual(m.timescales, (60.0,))

    def test_write_and_read(self):
        m = ListEpisodicMemory()
        ctx = MemoryContext(timestamp=10.0)
        e = MemoryEntry(
            "e1",
            "saw a red chair",
            semantic_keys=["chair", "red"],
            spatial_keys=[(1.0, 2.0)],
            temporal_keys=[10.0],
        )
        self.assertTrue(m.write(e, ctx))
        result = m.read(Query())
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].entry_id, "e1")
        self.assertEqual(result.source_slot, "episodic")

    def test_write_synthesises_temporal_key_if_absent(self):
        m = ListEpisodicMemory()
        e = MemoryEntry("e1", "no time")
        m.write(e, MemoryContext())
        # Now retrievable via timeline.
        timeline = m.get_timeline()
        self.assertEqual(len(timeline), 1)
        self.assertEqual(len(timeline[0].temporal_keys), 1)

    def test_capacity_per_bucket_fifo_drops(self):
        m = ListEpisodicMemory(
            timescales=(60.0,), capacity_per_bucket=2
        )
        for i in range(5):
            m.write(MemoryEntry(f"e{i}", f"ev{i}", temporal_keys=[float(i)]),
                    MemoryContext())
        # Only the last 2 should remain in the bucket.
        self.assertEqual(m.stats()["count"], 5)  # _by_id still has all 5
        # But bucket size is enforced.
        self.assertEqual(m.stats()["per_bucket"]["60.0"], 2)

    def test_clear(self):
        m = ListEpisodicMemory()
        m.write(MemoryEntry("e1", "x", temporal_keys=[1.0]), MemoryContext())
        m.clear()
        self.assertEqual(m.stats()["count"], 0)
        self.assertEqual(m.read(Query()).entries, [])


class TestListEpisodicMultiAxis(unittest.TestCase):
    def setUp(self):
        self.m = ListEpisodicMemory()
        ctx = MemoryContext()
        self.m.write(MemoryEntry(
            "e1", "red chair near window",
            semantic_keys=["chair", "red"],
            spatial_keys=[(1.0, 2.0)],
            temporal_keys=[10.0],
        ), ctx)
        self.m.write(MemoryEntry(
            "e2", "blue sofa",
            semantic_keys=["sofa", "blue"],
            spatial_keys=[(3.0, 4.0)],
            temporal_keys=[20.0],
        ), ctx)
        self.m.write(MemoryEntry(
            "e3", "red sofa",
            semantic_keys=["sofa", "red"],
            spatial_keys=[(1.0, 2.0)],
            temporal_keys=[15.0],
        ), ctx)

    def test_semantic_query(self):
        r = self.m.read(QueryBuilder().with_semantic("chair").build())
        self.assertEqual({e.entry_id for e in r.entries}, {"e1"})

    def test_semantic_intersect(self):
        # red AND sofa -> e3
        r = self.m.read(QueryBuilder().with_semantic("red", "sofa").build())
        self.assertEqual({e.entry_id for e in r.entries}, {"e3"})

    def test_spatial_query(self):
        r = self.m.read(QueryBuilder().with_spatial((1.0, 2.0)).build())
        self.assertEqual({e.entry_id for e in r.entries}, {"e1", "e3"})

    def test_temporal_range_query(self):
        r = self.m.read(QueryBuilder().with_temporal(12, 18).build())
        self.assertEqual({e.entry_id for e in r.entries}, {"e3"})

    def test_combined_axes_intersection(self):
        # red AND at (1,2) AND temporal [0, 100]
        q = (
            QueryBuilder()
            .with_semantic("red")
            .with_spatial((1.0, 2.0))
            .with_temporal(0, 100)
            .build()
        )
        r = self.m.read(q)
        self.assertEqual({e.entry_id for e in r.entries}, {"e1", "e3"})

    def test_top_k_truncation(self):
        r = self.m.read(QueryBuilder().with_top_k(1).build())
        self.assertEqual(len(r.entries), 1)

    def test_empty_query_returns_all(self):
        r = self.m.read(Query())
        self.assertEqual(len(r.entries), 3)


class TestListEpisodicTimeline(unittest.TestCase):
    def test_get_timeline_sorted_by_time(self):
        m = ListEpisodicMemory()
        for i in [3, 1, 2, 5, 4]:
            m.write(MemoryEntry(f"e{i}", "x", temporal_keys=[float(i)]),
                    MemoryContext())
        timeline = m.get_timeline()
        times = [min(e.temporal_keys) for e in timeline]
        self.assertEqual(times, sorted(times))
        self.assertEqual(timeline[0].entry_id, "e1")

    def test_get_timeline_range(self):
        m = ListEpisodicMemory()
        for i in range(10):
            m.write(MemoryEntry(f"e{i}", "x", temporal_keys=[float(i)]),
                    MemoryContext())
        in_range = m.get_timeline(3, 5)
        self.assertEqual({e.entry_id for e in in_range}, {"e3", "e4", "e5"})


class TestListEpisodicConsolidation(unittest.TestCase):
    def test_default_consolidate_returns_bucket_summaries(self):
        m = ListEpisodicMemory(timescales=(60.0, 600.0))
        for i in range(3):
            m.write(MemoryEntry(f"e{i}", f"x{i}", temporal_keys=[float(i)]),
                    MemoryContext())
        summaries = m.consolidate(_Dummy(), MemoryContext())
        # Both buckets are non-empty.
        self.assertEqual(len(summaries), 2)
        for s in summaries:
            self.assertEqual(s.source_slot, "episodic")
            self.assertEqual(s.metadata["kind"], "bucket_summary")

    def test_drop_oldest(self):
        m = ListEpisodicMemory()
        for i in range(5):
            m.write(MemoryEntry(f"e{i}", "x", temporal_keys=[float(i)]),
                    MemoryContext())
        n = m.drop_oldest(2)
        self.assertEqual(n, 2)
        remaining = {e.entry_id for e in m.read(Query()).entries}
        self.assertEqual(remaining, {"e2", "e3", "e4"})

    def test_drop_oldest_zero(self):
        m = ListEpisodicMemory()
        m.write(MemoryEntry("e", "x", temporal_keys=[1.0]), MemoryContext())
        self.assertEqual(m.drop_oldest(0), 0)


class _Dummy(MemoryModule):
    def __init__(self):
        super().__init__(slot=MemorySlot.WM)

    def write(self, e, c): return True
    def read(self, q): return QueryResult()
    def clear(self): pass
    def stats(self): return {}


# --------------------------------------------------------------------------- #
# FIFOForgetPolicy
# --------------------------------------------------------------------------- #
class TestFIFOForgetPolicy(unittest.TestCase):
    def test_drops_oldest_to_meet_capacity(self):
        m = ListEpisodicMemory()
        for i in range(5):
            m.write(MemoryEntry(f"e{i}", "x", temporal_keys=[float(i)]),
                    MemoryContext())
        pol = FIFOForgetPolicy(capacity=2)
        n = pol.apply(m, MemoryContext())
        self.assertEqual(n, 3)
        remaining = {e.entry_id for e in m.read(Query()).entries}
        self.assertEqual(remaining, {"e3", "e4"})

    def test_no_drop_when_under_capacity(self):
        m = ListEpisodicMemory()
        m.write(MemoryEntry("e", "x", temporal_keys=[1.0]), MemoryContext())
        self.assertEqual(FIFOForgetPolicy(capacity=5).apply(m, MemoryContext()), 0)

    def test_capacity_zero_drops_everything(self):
        m = ListEpisodicMemory()
        for i in range(3):
            m.write(MemoryEntry(f"e{i}", "x", temporal_keys=[float(i)]),
                    MemoryContext())
        n = FIFOForgetPolicy(capacity=0).apply(m, MemoryContext())
        self.assertEqual(n, 3)

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            FIFOForgetPolicy(capacity=-1)

    def test_safe_on_module_without_drop_oldest(self):
        pol = FIFOForgetPolicy(capacity=0)
        # _Dummy has no drop_oldest, so this should be a no-op
        self.assertEqual(pol.apply(_Dummy(), MemoryContext()), 0)


# --------------------------------------------------------------------------- #
# ExtractFactsConsolidationPolicy
# --------------------------------------------------------------------------- #
class _SimpleSM(SemanticMemoryABC):
    """Tiny semantic memory that just records written facts as triples."""

    def __init__(self):
        self._entries = []
        self._triples = []

    def write(self, e, c):
        self._entries.append(e)
        # Reconstruct a triple from metadata if present
        md = e.metadata
        if "subject" in md:
            self._triples.append((md["subject"], "was_seen", md.get("time_window")))
        return True

    def read(self, q):
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()
        self._triples.clear()

    def stats(self):
        return {"count": len(self._entries)}

    def add_fact(self, s, p, o):
        self._triples.append((s, p, o))
        return True

    def query_facts(self, s=None, p=None, obj=None):
        out = []
        for (ss, pp, oo) in self._triples:
            if s is not None and ss != s:
                continue
            if p is not None and pp != p:
                continue
            if obj is not None and oo != obj:
                continue
            out.append((ss, pp, oo))
        return out


class TestExtractFactsPolicy(unittest.TestCase):
    def test_extracts_one_fact_per_semantic_event(self):
        em = ListEpisodicMemory()
        em.write(MemoryEntry(
            "e1", "saw a red chair",
            semantic_keys=["chair"], temporal_keys=[10.0],
        ), MemoryContext())
        em.write(MemoryEntry(
            "e2", "saw a blue sofa",
            semantic_keys=["sofa"], temporal_keys=[20.0],
        ), MemoryContext())
        em.write(MemoryEntry(
            "e3", "no semantic tag",
            temporal_keys=[30.0],
        ), MemoryContext())
        sm = _SimpleSM()
        pol = ExtractFactsConsolidationPolicy()
        facts = pol.extract(em, sm, MemoryContext())
        # only e1 and e2 have semantic keys -> 2 facts
        self.assertEqual(len(facts), 2)
        subjects = sorted(f.metadata["subject"] for f in facts)
        self.assertEqual(subjects, ["chair", "sofa"])

    def test_extract_includes_all_events_when_only_with_semantic_false(self):
        em = ListEpisodicMemory()
        em.write(MemoryEntry("e1", "x", semantic_keys=["a"]), MemoryContext())
        em.write(MemoryEntry("e2", "y"), MemoryContext())
        pol = ExtractFactsConsolidationPolicy(only_with_semantic=False)
        facts = pol.extract(em, _SimpleSM(), MemoryContext())
        self.assertEqual(len(facts), 2)

    def test_custom_time_window(self):
        em = ListEpisodicMemory()
        em.write(MemoryEntry("e1", "x", semantic_keys=["a"], temporal_keys=[1.0]),
                 MemoryContext())
        pol = ExtractFactsConsolidationPolicy(time_window=42.0)
        facts = pol.extract(em, _SimpleSM(), MemoryContext())
        self.assertEqual(facts[0].metadata["time_window"], 42.0)

    def test_extracted_fact_ids_unique(self):
        em = ListEpisodicMemory()
        for i in range(5):
            em.write(MemoryEntry(f"e{i}", "x", semantic_keys=[f"k{i}"]),
                     MemoryContext())
        pol = ExtractFactsConsolidationPolicy()
        facts = pol.extract(em, _SimpleSM(), MemoryContext())
        ids = [f.entry_id for f in facts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_extract_then_write_into_sm_via_graph(self):
        # End-to-end: use MemoryGraph to wire EM -> SM and run a consolidation pass.
        from unimem.graph.edge import EdgeKind, MemoryEdge
        from unimem.graph.graph import MemoryGraph
        from unimem.graph.node import MemoryNode

        g = MemoryGraph()
        em = ListEpisodicMemory()
        sm = _SimpleSM()
        g.add_node(MemoryNode(node_id="em", slot=MemorySlot.EM, module=em))
        g.add_node(MemoryNode(node_id="sm", slot=MemorySlot.SM, module=sm))
        g.add_edge(MemoryEdge(
            "em", "sm", EdgeKind.CONSOLIDATES_TO,
            policy=ExtractFactsConsolidationPolicy(),
        ))
        # Stage events
        em.write(MemoryEntry(
            "e1", "saw a red chair", semantic_keys=["chair"], temporal_keys=[1.0],
        ), MemoryContext())
        em.write(MemoryEntry(
            "e2", "saw a sofa", semantic_keys=["sofa"], temporal_keys=[2.0],
        ), MemoryContext())
        # Run consolidation
        result = g.run_consolidation_pass(MemoryContext())
        self.assertEqual(result["stored"]["sm"], 2)
        # SM has both facts
        facts = sm.query_facts()
        subjects = sorted(s for (s, _, _) in facts)
        self.assertEqual(subjects, ["chair", "sofa"])


if __name__ == "__main__":
    unittest.main()
