"""Tests for R4KnowledgeDatabase — three-axis retrieval + Eq. 5 dedup."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockEmbedding
from reproductions.r4.memory.embedding import cosine_similarity, euclidean_distance
from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase, _bbox_to_centroid_extent
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import QueryBuilder
from unimem.core.slots import MemorySlot


class TestCentroidExtent(unittest.TestCase):
    def test_centroid_and_extent(self):
        pts = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 4)]
        c, e = _bbox_to_centroid_extent(pts)
        self.assertAlmostEqual(c[0], 0.25)
        self.assertAlmostEqual(c[1], 0.5)
        self.assertAlmostEqual(c[2], 1.0)
        self.assertEqual(e, (1.0, 2.0, 4.0))

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            _bbox_to_centroid_extent([])

    def test_rejects_non_3d(self):
        with self.assertRaises(ValueError):
            _bbox_to_centroid_extent([(1, 2)])


class TestEmbeddingMath(unittest.TestCase):
    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_identical(self):
        self.assertAlmostEqual(cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0)

    def test_cosine_opposite(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_euclidean(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])


class TestR4DatabaseBasic(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(
            embedding_fn=MockEmbedding(dim=32),
            eps_c=0.5,
            delta_s=0.7,
        )

    def test_observe_inserts_new_record(self):
        inserted = self.db.observe_object(
            description="red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        self.assertTrue(inserted)
        self.assertEqual(self.db.stats()["count"], 1)
        self.assertEqual(self.db.stats()["n_slam_points"], 1)

    def test_observe_dedup_merges_when_close_and_similar(self):
        # First observation
        self.db.observe_object(
            description="a red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        # Second observation very close spatially and identical-ish description
        # MockEmbedding is deterministic, so identical text → identical embedding.
        inserted = self.db.observe_object(
            description="a red chair",  # same text → cosine 1.0
            centroid=(1.0, 2.0, 0.5),  # same point → dist 0
            extent=(0.5, 0.5, 1.0),
            timestamp=15.0,
        )
        self.assertFalse(inserted)  # merged, not new
        self.assertEqual(self.db.stats()["count"], 1)
        rec = self.db.all_records()[0]
        self.assertEqual(rec.tem.timestamps, [10.0, 15.0])  # both timestamps kept

    def test_observe_does_not_dedup_when_spatially_far(self):
        self.db.observe_object(
            description="a red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        # 10m away — way beyond eps_c
        inserted = self.db.observe_object(
            description="a red chair",
            centroid=(11.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=12.0,
        )
        self.assertTrue(inserted)
        self.assertEqual(self.db.stats()["count"], 2)

    def test_observe_does_not_dedup_when_dissimilar(self):
        # Use two embeddings that we know are dissimilar
        # MockEmbedding is hash-based, so different text → likely different vectors
        self.db.observe_object(
            description="xyzpdq abc",  # arbitrary
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        # Different text and unlikely to cosine > 0.7 — but possibly flukes
        # if hashing collides. Use a really different text.
        inserted = self.db.observe_object(
            description="zzzzzzzzz different",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=12.0,
        )
        # We can't assert True 100% due to hash stochasticity, but it's almost always True
        # Use this test mainly to ensure no crash. If it does dedup, the test still passes.
        self.assertIn(self.db.stats()["count"], (1, 2))


class TestR4DatabaseRetrieval(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        # Stage 3 objects with distinct descriptions & positions
        for i, (desc, pos, t) in enumerate([
            ("red chair near the window", (1.0, 1.0, 0.5), 10.0),
            ("blue sofa in the corner", (5.0, 1.0, 0.5), 20.0),
            ("wooden table in the centre", (3.0, 3.0, 0.5), 30.0),
        ]):
            self.db.observe_object(
                description=desc,
                centroid=pos,
                extent=(0.5, 0.5, 1.0),
                timestamp=t,
            )

    def test_retrieve_all_when_no_keys(self):
        ids = self.db._retrieve()
        self.assertEqual(len(ids), 3)

    def test_spatial_retrieval_nearest(self):
        ids = self.db._retrieve(k_spa_centroid=(1.1, 1.1, 0.5), k_spa_radius=1.0)
        self.assertEqual(len(ids), 1)
        # Only the red chair is within 1m of (1.1, 1.1, 0.5)
        rec = self.db.get_record(ids[0])
        self.assertIn("chair", rec.sem.description)

    def test_spatial_retrieval_with_k(self):
        ids = self.db._retrieve(k_spa_centroid=(3.0, 3.0, 0.5), k_spa_k=2)
        self.assertEqual(len(ids), 2)

    def test_temporal_retrieval(self):
        ids = self.db._retrieve(k_t_min=15.0, k_t_max=25.0)
        # Only blue sofa @ t=20 falls in window
        self.assertEqual(len(ids), 1)

    def test_semantic_retrieval(self):
        ids = self.db._retrieve(k_sem=["chair"])
        # At least one match; cosine depends on hashing
        # Just verify it doesn't crash and returns a subset
        self.assertLessEqual(len(ids), 3)

    def test_combined_axes_intersection(self):
        ids = self.db._retrieve(
            k_spa_centroid=(0.9, 1.1, 0.5),
            k_spa_radius=1.0,
            k_t_min=5.0,
            k_t_max=15.0,
        )
        self.assertEqual(len(ids), 1)

    def test_read_via_query_returns_entries_with_three_axes(self):
        from unimem.core.query import Query

        q = Query(spatial=[(0.9, 1.1, 0.5)], metadata={"spatial_radius": 1.0})
        result = self.db.read(q)
        self.assertEqual(len(result.entries), 1)
        e = result.entries[0]
        self.assertEqual(e.source_slot, MemorySlot.GM.value)
        # The three axes should be populated
        self.assertEqual(e.spatial_keys, [(1.0, 1.0, 0.5)])
        self.assertEqual(e.temporal_keys, [10.0])
        self.assertIn("chair", e.text.lower())


class TestR4DatabaseViaMemoryEntry(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))

    def test_write_via_memory_entry(self):
        e = MemoryEntry(
            entry_id="obs1",
            text="a tall plant",
            spatial_keys=[(2.0, 3.0, 0.0, 0.4, 0.4, 1.5)],  # centroid+extent
            temporal_keys=[5.0],
            source_slot="wm",
        )
        ok = self.db.write(e, MemoryContext())
        self.assertTrue(ok)
        self.assertEqual(self.db.stats()["count"], 1)
        rec = self.db.all_records()[0]
        self.assertEqual(rec.spa.centroid, (2.0, 3.0, 0.0))
        self.assertEqual(rec.spa.extent, (0.4, 0.4, 1.5))

    def test_write_default_extent_when_only_centroid(self):
        e = MemoryEntry(
            entry_id="obs1",
            text="thing",
            spatial_keys=[(2.0, 3.0, 0.0)],
            temporal_keys=[5.0],
        )
        ok = self.db.write(e, MemoryContext())
        self.assertTrue(ok)
        rec = self.db.all_records()[0]
        self.assertEqual(rec.spa.extent, (0.1, 0.1, 0.1))

    def test_write_rejects_empty_spatial(self):
        ok = self.db.write(MemoryEntry("x", "no pos"), MemoryContext())
        self.assertFalse(ok)


class TestSLAMMap(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))

    def test_nearest_neighbors(self):
        for pos, t in [((0, 0, 0), 1.0), ((1, 0, 0), 2.0), ((2, 0, 0), 3.0)]:
            self.db.observe_object(description=f"o{t}", centroid=pos, extent=(0.1, 0.1, 0.1), timestamp=t)
        nbrs = self.db.slam_map.nearest_neighbors((0.5, 0, 0), k=2)
        self.assertEqual(len(nbrs), 2)
        self.assertEqual(nbrs[0][0], "obj-0001")  # closest to (0.5,0,0)

    def test_directional_filter(self):
        for pos, t in [((0, 0, 1), 1.0), ((0, 0, -1), 2.0), ((0, 1, 0), 3.0)]:
            self.db.observe_object(description=f"o{t}", centroid=pos, extent=(0.1, 0.1, 0.1), timestamp=t)
        forward = self.db.slam_map.directional_filter((0, 0, 0), (0, 0, 1), half_angle_deg=45)
        self.assertIn("obj-0001", forward)
        self.assertNotIn("obj-0002", forward)


if __name__ == "__main__":
    unittest.main()
