"""Cross-method integration tests.

Validates that all three reproductions (R4, CLiViS, VideoHV) plug into the
same unimem framework via the standard ``MemoryModule`` contract, and that
the framework's graph algorithms work uniformly across them.

Specifically:

* Every method's memory modules are real ``MemoryModule`` subclasses.
* Each method's graph builds and exposes the expected topology.
* The framework's three core algorithms (read fan-out, write fan-in,
  consolidation) work end-to-end on each method.
* A mocked end-to-end scenario runs for each method, demonstrating that
  the framework is genuinely method-agnostic.
"""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockEmbedding, MockLLM, MockVLM
from reproductions.clivis import (
    NavigationGraph,
    RelationGraph,
    TimeWorkingMemory,
)
from reproductions.clivis.graph_spec import build_clivis_graph
from reproductions.r4.graph_spec import build_r4_graph
from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase
from reproductions.videohv.graph_spec import build_videohv_graph
from reproductions.videohv.memory.time_verification_trace import VerificationTraceMemory
from reproductions.videohv.memory.video_summary_memory import VideoSummaryMemory
from unimem.core.module import MemoryModule
from unimem.core.slots import MemorySlot


# --------------------------------------------------------------------------- #
# All methods implement MemoryModule
# --------------------------------------------------------------------------- #
class TestAllModulesAreMemoryModules(unittest.TestCase):
    def _check(self, mod):
        self.assertIsInstance(mod, MemoryModule)
        # Required abstract methods are all implemented
        self.assertFalse(getattr(mod.__class__, "__abstractmethods__", set()),
                         f"{type(mod).__name__} has unimplemented abstracts")

    def test_r4_db(self):
        self._check(R4KnowledgeDatabase(embedding_fn=MockEmbedding()))

    def test_clivis_memories(self):
        self._check(TimeWorkingMemory(question="q"))
        self._check(NavigationGraph())
        self._check(RelationGraph())

    def test_videohv_memories(self):
        self._check(VideoSummaryMemory(action_captions=["a"]))
        self._check(VerificationTraceMemory())


# --------------------------------------------------------------------------- #
# All graphs build via the same framework API
# --------------------------------------------------------------------------- #
class TestGraphConstruction(unittest.TestCase):
    def test_r4_graph_topology(self):
        g = build_r4_graph(embedding_fn=MockEmbedding())
        self.assertEqual(len(g), 3)
        # WM → DB → SM
        self.assertEqual(g.get_node("wm").slot, MemorySlot.WM)
        self.assertEqual(g.get_node("db").slot, MemorySlot.GM)
        self.assertEqual(g.get_node("sm").slot, MemorySlot.SM)
        kinds = sorted(e.kind.name for e in g.edges())
        self.assertEqual(kinds, ["CONSOLIDATES_TO", "FEEDS"])

    def test_clivis_graph_topology(self):
        g = build_clivis_graph(question="q")
        self.assertEqual(len(g), 3)
        # WM + SG + GM
        self.assertEqual(g.get_node("wm").slot, MemorySlot.WM)
        self.assertEqual(g.get_node("sg").slot, MemorySlot.SG)
        self.assertEqual(g.get_node("gm").slot, MemorySlot.GM)
        kinds = sorted(e.kind.name for e in g.edges())
        # 2 FEEDS (wm→sg, wm→gm) + 1 INDEXES (sg→gm)
        self.assertEqual(kinds, ["FEEDS", "FEEDS", "INDEXES"])

    def test_videohv_graph_topology(self):
        g = build_videohv_graph()
        self.assertEqual(len(g), 2)
        # Both slots are EM (two episodic memories)
        self.assertEqual(g.get_node("summary").slot, MemorySlot.EM)
        self.assertEqual(g.get_node("trace").slot, MemorySlot.EM)
        kinds = [e.kind.name for e in g.edges()]
        self.assertEqual(kinds, ["REFERENCES"])


# --------------------------------------------------------------------------- #
# Fan-in write works on each method's graph
# --------------------------------------------------------------------------- #
class TestFanInWrite(unittest.TestCase):
    def test_r4_observation_propagates_wm_to_db(self):
        from unimem.core.context import MemoryContext
        from unimem.core.entry import MemoryEntry

        g = build_r4_graph(embedding_fn=MockEmbedding())
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
        # The DB now has the record
        db = g.get_node("db").module
        self.assertEqual(db.stats()["count"], 1)

    def test_clivis_observation_propagates_wm_to_sg_and_gm(self):
        from unimem.core.context import MemoryContext
        from unimem.core.entry import MemoryEntry

        g = build_clivis_graph(question="q")
        # Pre-seed a period so the object write can land somewhere
        g.get_node("gm").module._init_period("00:00:00-00:00:30", "")
        obs = MemoryEntry(
            entry_id="obs1",
            text="a cup",
            metadata={"kind": "object", "name": "cup", "period": "00:00:00-00:00:30"},
        )
        result = g.write(obs, MemoryContext(), source_node_id="wm")
        self.assertTrue(result["wm"])
        self.assertTrue(result["sg"])
        self.assertTrue(result["gm"])
        # SG has the cup as an Object node
        self.assertIsNotNone(g.get_node("sg").module.get_node_info("cup"))
        # GM has the cup registered in the period
        self.assertIn("cup", g.get_node("gm").module.periods_to_obj_names["00:00:00-00:00:30"])

    def test_videohv_writes_dont_propagate(self):
        """VideoHV's two memories are independent (REFERENCES, not FEEDS)."""
        from unimem.core.context import MemoryContext
        from unimem.core.entry import MemoryEntry

        g = build_videohv_graph()
        obs = MemoryEntry("obs1", "summary entry")
        result = g.write(obs, MemoryContext(), source_node_id="summary")
        # Only the source node should have been reached
        self.assertIn("summary", result)
        # No propagation to trace (REFERENCES doesn't propagate writes)
        self.assertNotIn("trace", result)


# --------------------------------------------------------------------------- #
# Fan-out read works on each method's graph
# --------------------------------------------------------------------------- #
class TestFanOutRead(unittest.TestCase):
    def test_r4_query_returns_db_entries(self):
        from unimem.core.query import QueryBuilder

        g = build_r4_graph(embedding_fn=MockEmbedding())
        db = g.get_node("db").module
        db.observe_object(
            description="red chair",
            centroid=(1.0, 1.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        # Query the GM slot
        results = g.read(QueryBuilder().with_slot(MemorySlot.GM).build())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_slot, "spatial_geometric")
        self.assertEqual(len(results[0].entries), 1)

    def test_clivis_query_returns_from_all_three_slots(self):
        from unimem.core.query import QueryBuilder

        g = build_clivis_graph(question="q")
        # Put something in each
        g.get_node("wm").module.append_event(_entry("r1"))
        g.get_node("sg").module.add_person("alice", "host")
        g.get_node("gm").module._init_period("p1", "x")
        # Query across all slots (no slot_filter)
        results = g.read(QueryBuilder().build())
        self.assertEqual(len(results), 3)

    def test_videohv_query_returns_summary_and_trace(self):
        from unimem.core.query import QueryBuilder

        g = build_videohv_graph()
        # Pre-load summaries
        g.get_node("summary").module.ingest(["clip A", "clip B"])
        g.get_node("trace").module.record_round(0, clue="c1")
        results = g.read(QueryBuilder().build())
        # Both memories should respond
        self.assertEqual(len(results), 2)
        per_slot = {r.source_node_id: len(r.entries) for r in results}
        self.assertEqual(per_slot["summary"], 2)
        self.assertEqual(per_slot["trace"], 1)


# --------------------------------------------------------------------------- #
# Consolidation pass works on R4 (the only method with CONSOLIDATES_TO)
# --------------------------------------------------------------------------- #
class TestConsolidation(unittest.TestCase):
    def test_r4_consolidation_runs_without_error(self):
        from unimem.core.context import MemoryContext

        g = build_r4_graph(embedding_fn=MockEmbedding())
        # The DB has a default consolidate() returning bucket summaries
        db = g.get_node("db").module
        # The default ObjectRecord has no timescale buckets → consolidate
        # returns []. Just make sure the pass runs without error.
        result = g.run_consolidation_pass(MemoryContext())
        self.assertIn("extracted", result)
        self.assertIn("stored", result)
        self.assertIn("forgotten", result)


# --------------------------------------------------------------------------- #
# End-to-end mocked scenarios
# --------------------------------------------------------------------------- #
def _entry(text):
    from unimem.core.entry import MemoryEntry
    return MemoryEntry(entry_id=text, text=text)


class TestEndToEndScenarios(unittest.TestCase):
    def test_r4_storage_then_retrieval(self):
        """Simulate: agent observes 3 objects, then asks 'where is the chair?'"""
        from reproductions.r4.pipeline import (
            Observation,
            R4Pipeline,
            SegmentedObject,
        )

        db = R4KnowledgeDatabase(embedding_fn=MockEmbedding())
        # VLM gives a deterministic description per call
        vlm = MockVLM(default_response="a piece of furniture")
        pipe = R4Pipeline(db=db, vlm=vlm)
        # 3 segmented objects with pre-filled descriptions
        pipe.store(
            Observation(timestamp=1.0, camera_pose=(0, 0, 0)),
            [
                SegmentedObject(mask_points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                                timestamp=1.0, description="red chair"),
                SegmentedObject(mask_points=[(5, 0, 0), (6, 0, 0), (5, 1, 0)],
                                timestamp=1.0, description="blue sofa"),
                SegmentedObject(mask_points=[(0, 0, 5), (1, 0, 5), (0, 1, 5)],
                                timestamp=1.0, description="wooden table"),
            ],
        )
        self.assertEqual(db.stats()["count"], 3)
        # Now retrieve by spatial proximity to (0.5, 0, 0)
        ids = db._retrieve(k_spa_centroid=(0.5, 0, 0), k_spa_radius=2.0)
        self.assertEqual(len(ids), 1)
        rec = db.get_record(ids[0])
        self.assertIn("chair", rec.sem.description)

    def test_clivis_full_loop_with_mocks(self):
        """Simulate: CLiViS iterates once and produces a final answer."""
        import json

        from reproductions.clivis.pipeline import CLiViSPipeline, PeriodInput

        # LLM script:
        # 1) init: return canned graph
        # 2) instruction round 0: return [final] answer immediately
        init_response = json.dumps({
            "persons": [{"name": "alice", "info": "host"}],
            "areas": [{"name": "kitchen", "info": "", "time_range": "00:00:00-00:00:30"}],
            "objects": [{"name": "cup", "period": "00:00:00-00:00:30"}],
            "relations": [],
            "actions": [],
        })
        llm = MockLLM(responses=[init_response, "[final] alice is in the kitchen"])
        vlm = MockVLM(default_response="ok")
        pipe = CLiViSPipeline(llm=llm, vlm=vlm, max_rounds=5)
        result = pipe.run(
            "Where is alice?",
            [PeriodInput("00:00:00-00:00:30", "kitchen scene", "/tmp/seg.mp4")],
        )
        self.assertEqual(result.answer, "alice is in the kitchen")
        self.assertEqual(result.n_rounds, 0)

    def test_videohv_verified_in_round_1(self):
        """Simulate: VideoHV verifies a clue and selects an answer."""
        from reproductions.videohv.pipeline import VideoHVBundle, VideoHVPipeline

        llm = MockLLM(responses=[
            "0: alice\n1: bob",      # hypotheses
            "0.9|the cup",           # distinctness
            "verified",              # verification
            "1",                     # answer
        ])
        pipe = VideoHVPipeline(llm=llm)
        bundle = VideoHVBundle(
            action_caption_summaries=["alice pouring coffee"],
            options=["alice", "bob"],
        )
        result = pipe.run("Who poured coffee?", bundle)
        self.assertTrue(result.verified)
        self.assertEqual(result.answer, 1)


# --------------------------------------------------------------------------- #
# Unified-storage integration — the key new property enabled by the
# storage-backed refactor: a single GraphStorage can host nodes from
# multiple methods at once, enabling cross-method consistency queries.
# --------------------------------------------------------------------------- #
class TestUnifiedStorage(unittest.TestCase):
    def test_shared_storage_across_methods(self):
        """All three methods can write into the same GraphStorage."""
        from unimem.graph_storage import InMemoryGraphStorage

        gs = InMemoryGraphStorage()

        # R4 facade
        r4_db = R4KnowledgeDatabase(
            embedding_fn=MockEmbedding(), graph_storage=gs
        )
        # CLiViS facades
        rel = RelationGraph(graph_storage=gs)
        nav = NavigationGraph(graph_storage=gs)
        wm = TimeWorkingMemory(question="q", graph_storage=gs)
        # VideoHV facades
        summary = VideoSummaryMemory(graph_storage=gs)
        trace = VerificationTraceMemory(graph_storage=gs)

        # Each writes its own data — no collisions
        r4_db.observe_object(
            description="red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        rel.add_person("alice", "host")
        nav.add_persons([{"name": "bob", "info": "guest"}])
        wm.update_history_msg("look", "I see alice")
        from reproductions._common.mocks import MockLLM
        wm._llm_extractor = MockLLM(default_response='{"evidence": "alice is here"}')
        wm.extract_and_update_rationale_list("00:00:00-00:00:30")
        summary.ingest(["clip A", "clip B"])
        trace.record_round(0, clue="c1")

        # All data sits in one storage backend
        node = gs.get_node("obj-0001")
        self.assertIsNotNone(node)
        node = gs.get_node("alice")
        self.assertIsNotNone(node)
        node = gs.get_node("rationale-0")
        self.assertIsNotNone(node)
        node = gs.get_node("clip-0")
        self.assertIsNotNone(node)
        node = gs.get_node("trace-round-0")
        self.assertIsNotNone(node)

    def test_cypher_can_query_across_methods(self):
        """A single Cypher query can return nodes from multiple methods."""
        from unimem.graph_storage import InMemoryGraphStorage

        gs = InMemoryGraphStorage()
        rel = RelationGraph(graph_storage=gs)
        summary = VideoSummaryMemory(graph_storage=gs)

        rel.add_person("alice", "host")
        summary.ingest(["alice enters", "alice sits"])

        # Query: every node in the storage
        rows = gs.query("MATCH (n) RETURN n")
        # Should include alice + clip-0 + clip-1 + their TimeIndex nodes
        node_ids = {r["n"]["node_id"] for r in rows}
        self.assertIn("alice", node_ids)
        self.assertIn("clip-0", node_ids)
        self.assertIn("clip-1", node_ids)

    def test_clivis_and_videohv_share_timeindex_pattern(self):
        """Both CLiViS rationales and VideoHV clips use :AT_TIME edges,
        so a single time-range query works across both."""
        from unimem.graph_storage import InMemoryGraphStorage

        gs = InMemoryGraphStorage()
        wm = TimeWorkingMemory(question="q", graph_storage=gs)
        summary = VideoSummaryMemory(graph_storage=gs)

        # CLiViS rationale with period 00:00:10-00:00:20
        wm.update_history_msg("look", "alice walks")
        from reproductions._common.mocks import MockLLM
        wm._llm_extractor = MockLLM(
            default_response='{"evidence": "alice walks", "related_area": "kitchen"}'
        )
        wm.extract_and_update_rationale_list("00:00:10-00:00:20")
        # VideoHV clip with start_t=15, end_t=20
        summary.ingest(
            ["alice walks"],
            clip_boundaries=[(15.0, 20.0)],
        )
        # Query TimeIndex nodes overlapping [12, 18]
        results = gs.get_time_indexed_nodes(time_range=(12.0, 18.0))
        result_ids = sorted({e.entry_id for e in results})
        # Should contain both the rationale and the clip
        self.assertIn("rationale-0", result_ids)
        self.assertIn("clip-0", result_ids)


if __name__ == "__main__":
    unittest.main()
