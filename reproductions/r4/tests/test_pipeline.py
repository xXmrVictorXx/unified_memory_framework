"""Tests for the R4 pipeline (storage + retrieval-augmented reasoning)."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockEmbedding, MockVLM
from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase
from reproductions.r4.pipeline import (
    Observation,
    R4Pipeline,
    SegmentedObject,
)


class TestStoragePass(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        self.pipe = R4Pipeline(db=self.db, vlm=MockVLM(default_response="auto-desc"))

    def test_store_inserts_records(self):
        obs = Observation(timestamp=1.0, camera_pose=(0.0, 0.0, 0.0))
        seg = SegmentedObject(
            mask_points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            timestamp=1.0,
            description="a box",  # pre-filled; no VLM call needed
        )
        n = self.pipe.store(obs, [seg])
        self.assertEqual(n, 1)
        self.assertEqual(self.db.stats()["count"], 1)
        # Pose was logged
        self.assertEqual(len(self.db.slam_map._trajectory), 1)

    def test_store_uses_vlm_when_no_description(self):
        obs = Observation(timestamp=1.0)
        seg = SegmentedObject(
            mask_points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            timestamp=1.0,
            description=None,  # VLM should be called
        )
        self.pipe.store(obs, [seg])
        # MockVLM should have been called once for description
        self.assertEqual(self.pipe.vlm.calls["__call__"].__len__(), 1)

    def test_store_dedup_merges(self):
        obs = Observation(timestamp=1.0)
        seg1 = SegmentedObject(
            mask_points=[(0, 0, 0), (1, 0, 0)], timestamp=1.0, description="red chair",
        )
        seg2 = SegmentedObject(
            mask_points=[(0, 0, 0), (1, 0, 0)], timestamp=2.0, description="red chair",
        )
        n1 = self.pipe.store(obs, [seg1])
        n2 = self.pipe.store(obs, [seg2])
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 0)  # merged
        self.assertEqual(self.db.stats()["count"], 1)
        rec = self.db.all_records()[0]
        self.assertEqual(rec.tem.timestamps, [1.0, 2.0])


class TestAnswerLoop(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        # Stage one record
        self.db.observe_object(
            description="red chair near window",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        self.db.observe_object(
            description="blue sofa in the corner",
            centroid=(5.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=20.0,
        )

    def test_stage1_short_circuits_on_confidence(self):
        vlm = MockVLM(default_response="[confident] The answer is 42.")
        pipe = R4Pipeline(db=self.db, vlm=vlm)
        result = pipe.answer("What is the meaning?")
        self.assertFalse(result.used_retrieval)
        self.assertEqual(result.answer, "The answer is 42.")
        self.assertEqual(result.n_retrieval_rounds, 0)

    def test_stage2_engages_retrieval_on_no_confidence(self):
        # First call says "no confident"; subsequent calls say "confident".
        vlm = MockVLM(
            responses=[
                "[need-retrieval]",
                "[confident] It's the red chair.",
            ]
        )
        pipe = R4Pipeline(db=self.db, vlm=vlm)
        result = pipe.answer("Where is the chair?")
        self.assertTrue(result.used_retrieval)
        self.assertEqual(result.n_retrieval_rounds, 1)
        self.assertEqual(result.answer, "It's the red chair.")

    def test_stage2_caps_at_max_rounds(self):
        # VLM always says "need retrieval" → exhausts max_rounds.
        vlm = MockVLM(default_response="[need-retrieval] still thinking")
        pipe = R4Pipeline(db=self.db, vlm=vlm)
        result = pipe.answer("Where?", max_rounds=2)
        self.assertEqual(result.n_retrieval_rounds, 2)
        self.assertTrue(result.used_retrieval)

    def test_retrieval_trace_recorded(self):
        vlm = MockVLM(responses=["[need-retrieval]", "[confident] ok"])
        pipe = R4Pipeline(db=self.db, vlm=vlm)
        result = pipe.answer("Where is the chair?")
        self.assertEqual(len(result.retrieval_trace), 1)
        trace = result.retrieval_trace[0]
        # Heuristic decomposer extracts significant words as k_sem (lower-cased)
        self.assertIn("chair", trace.k_sem)
        self.assertIn("where", trace.k_sem)


class TestHeuristicDecomposer(unittest.TestCase):
    def test_extracts_semantic_tokens(self):
        from reproductions.r4.pipeline import _heuristic_decomposer

        keys = _heuristic_decomposer("Where is the red chair?", None)
        self.assertIn("chair", keys["k_sem"])
        self.assertIn("where", keys["k_sem"])
        # Words <=4 chars are dropped
        self.assertNotIn("the", keys["k_sem"])
        self.assertNotIn("red", keys["k_sem"])

    def test_parses_seconds_ago(self):
        from reproductions.r4.pipeline import _heuristic_decomposer

        keys = _heuristic_decomposer("What did I see 5 seconds ago?", None)
        self.assertEqual(keys["k_t_min"], -5.0)


if __name__ == "__main__":
    unittest.main()
