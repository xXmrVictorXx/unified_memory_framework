"""Tests for VideoHV synthesised memory modules (storage-backed)."""
from __future__ import annotations

import unittest

from reproductions.videohv.memory.time_verification_trace import VerificationTraceMemory
from reproductions.videohv.memory.video_summary_memory import VideoSummaryMemory
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import QueryBuilder
from unimem.core.slots import MemorySlot


class TestVideoSummaryMemory(unittest.TestCase):
    def setUp(self):
        self.mem = VideoSummaryMemory(
            action_captions=[
                "person opening the fridge",
                "person pouring coffee",
                "person sitting on the sofa",
            ],
            object_detections=[
                [{"label": "fridge"}, {"label": "person"}],
                [{"label": "cup"}, {"label": "person"}],
                [{"label": "sofa"}, {"label": "person"}],
            ],
            clip_boundaries=[(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)],
        )

    def test_is_memory_module(self):
        # No longer EpisodicMemoryABC — just a plain MemoryModule subclass.
        self.assertIsInstance(self.mem, MemoryModule)
        self.assertEqual(self.mem.slot, MemorySlot.EM)

    def test_ingest_count(self):
        self.assertEqual(self.mem.stats()["count"], 3)

    def test_clip_entries_have_semantic_keys(self):
        clip0 = self.mem.get_clip(0)
        self.assertIsNotNone(clip0)
        self.assertIn("fridge", clip0.semantic_keys)
        self.assertIn("person", clip0.semantic_keys)
        self.assertIn("clip-0", clip0.semantic_keys)

    def test_temporal_keys(self):
        clip1 = self.mem.get_clip(1)
        self.assertEqual(clip1.temporal_keys, [5.0, 10.0])

    def test_get_timeline_returns_sorted(self):
        timeline = self.mem.get_timeline()
        self.assertEqual(len(timeline), 3)
        starts = [e.metadata["start_t"] for e in timeline]
        self.assertEqual(starts, sorted(starts))

    def test_get_timeline_temporal_filter(self):
        timeline = self.mem.get_timeline(6.0, 11.0)
        self.assertEqual({e.entry_id for e in timeline}, {"clip-1", "clip-2"})

    def test_read_by_semantic_object_tag(self):
        result = self.mem.read(QueryBuilder().with_semantic("person").build())
        self.assertEqual(len(result.entries), 3)
        result = self.mem.read(QueryBuilder().with_semantic("fridge").build())
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].entry_id, "clip-0")

    def test_read_by_clip_id(self):
        result = self.mem.read(QueryBuilder().with_semantic("clip-1").build())
        self.assertEqual(len(result.entries), 1)

    def test_read_combined_axes(self):
        result = self.mem.read(
            QueryBuilder()
            .with_semantic("person", "fridge")
            .with_temporal(0, 5)
            .build()
        )
        self.assertEqual(len(result.entries), 1)

    def test_object_tags_extracted(self):
        tags = self.mem.get_object_tags()
        self.assertEqual(sorted(tags), ["cup", "fridge", "person", "sofa"])

    def test_stats_top_tags(self):
        s = self.mem.stats()
        self.assertIn("top_tags", s)
        top = dict(s["top_tags"])
        self.assertEqual(top.get("person"), 3)

    def test_clear(self):
        self.mem.clear()
        self.assertEqual(self.mem.stats()["count"], 0)

    def test_default_clip_boundaries_when_missing(self):
        m = VideoSummaryMemory(action_captions=["a", "b"])
        self.assertEqual(m.get_clip(0).temporal_keys, [0.0, 1.0])
        self.assertEqual(m.get_clip(1).temporal_keys, [1.0, 2.0])


class TestVideoSummaryWriteViaEntry(unittest.TestCase):
    def test_write_adds_entry(self):
        m = VideoSummaryMemory()
        e = MemoryEntry(
            "clip-99", "manual clip",
            semantic_keys=["custom"],
            temporal_keys=[100.0, 110.0],
            metadata={"clip_index": 99},
        )
        self.assertTrue(m.write(e, MemoryContext()))
        result = m.read(QueryBuilder().with_semantic("custom").build())
        self.assertEqual(len(result.entries), 1)


class TestVideoSummaryStorage(unittest.TestCase):
    def setUp(self):
        self.mem = VideoSummaryMemory(
            action_captions=["a", "b"],
            object_detections=[[], []],
            clip_boundaries=[(0.0, 1.0), (1.0, 2.0)],
        )

    def test_clip_nodes_persisted_with_labels(self):
        node = self.mem.graph_storage.get_node("clip-0")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.EM.value, node["labels"])
        self.assertIn("VideoClip", node["labels"])

    def test_time_index_attached_per_clip(self):
        neighbours = self.mem.graph_storage.get_neighbors("clip-0", "AT_TIME", "out")
        self.assertEqual(len(neighbours), 1)
        ti = self.mem.graph_storage.get_node(neighbours[0][0])
        self.assertEqual(ti["properties"]["clip_index"], 0)
        self.assertEqual(ti["properties"]["start_t"], 0.0)
        self.assertEqual(ti["properties"]["end_t"], 1.0)

    def test_query_by_clip_index_via_storage(self):
        # Native Cypher-style query at the storage level.
        results = self.mem.graph_storage.get_time_indexed_nodes(clip_index=1)
        self.assertEqual([e.entry_id for e in results], ["clip-1"])


class TestVerificationTraceMemory(unittest.TestCase):
    def test_record_and_get_round(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, hypotheses=["h1", "h2"], clue="the cup", distinction_score=0.8)
        tm.record_round(0, verdict="verified")
        r = tm.get_round(0)
        self.assertEqual(r["hypotheses"], ["h1", "h2"])
        self.assertEqual(r["clue"], "the cup")
        self.assertEqual(r["verdict"], "verified")
        self.assertEqual(r["distinction_score"], 0.8)

    def test_n_rounds(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, clue="a")
        tm.record_round(1, clue="b")
        tm.record_round(2, clue="c")
        self.assertEqual(tm.n_rounds, 3)

    def test_all_rounds_returns_copies(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, clue="a")
        rounds = tm.all_rounds()
        rounds[0]["clue"] = "modified"
        self.assertEqual(tm.get_round(0)["clue"], "a")

    def test_is_memory_module(self):
        self.assertIsInstance(VerificationTraceMemory(), MemoryModule)

    def test_get_timeline_filters_by_round(self):
        tm = VerificationTraceMemory()
        for i in range(3):
            tm.record_round(i, clue=f"c{i}")
        timeline = tm.get_timeline(t_min=1, t_max=2)
        self.assertEqual({e.metadata["round"] for e in timeline}, {1, 2})

    def test_read_filters_by_semantic(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, verdict="verified")
        tm.record_round(1, verdict="not_verified")
        result = tm.read(QueryBuilder().with_semantic("verified").build())
        ids = {e.entry_id for e in result.entries}
        self.assertIn("trace-round-0", ids)

    def test_clear(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, clue="a")
        tm.clear()
        self.assertEqual(tm.stats()["count"], 0)

    def test_write_via_memory_entry(self):
        tm = VerificationTraceMemory()
        e = MemoryEntry(
            "x", "y",
            metadata={
                "round": 5,
                "clue": "test clue",
                "verdict": "verified",
            },
        )
        tm.write(e, MemoryContext())
        r = tm.get_round(5)
        self.assertIsNotNone(r)
        self.assertEqual(r["clue"], "test clue")


class TestTraceStorage(unittest.TestCase):
    def test_trace_nodes_have_time_index(self):
        tm = VerificationTraceMemory()
        tm.record_round(0, clue="first")
        tm.record_round(1, clue="second")
        neighbours = tm.graph_storage.get_neighbors("trace-round-1", "AT_TIME", "out")
        self.assertEqual(len(neighbours), 1)
        ti = tm.graph_storage.get_node(neighbours[0][0])
        self.assertEqual(ti["properties"]["timestamp"], 1.0)


if __name__ == "__main__":
    unittest.main()
