"""Tests verifying the storage-backed R4 rewrite uses DedupPolicy + GraphStorage + VectorStorage."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockEmbedding
from reproductions.r4.memory.knowledge_db import (
    R4_OBJECT_LABEL,
    R4KnowledgeDatabase,
)
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.slots import MemorySlot
from unimem.policies.write_policy import DedupPolicy


class TestR4IsStorageBacked(unittest.TestCase):
    def test_db_exposes_storage_backends(self):
        db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        # The new R4KnowledgeDatabase is a facade over a MemoryModule + storages.
        self.assertIsNotNone(db.graph_storage)
        self.assertIsNotNone(db.vector_storage)
        self.assertIsInstance(db.module, MemoryModule)
        self.assertIsInstance(db.module.write_policy, DedupPolicy)

    def test_module_uses_gm_slot(self):
        db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        self.assertEqual(db.module.slot, MemorySlot.GM)


class TestR4StorageLayout(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))

    def test_object_nodes_get_r4_label(self):
        self.db.observe_object(
            description="red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        rows = self.db.graph_storage.query(
            f"MATCH (n:{R4_OBJECT_LABEL}) RETURN n.node_id AS id"
        )
        self.assertEqual(len(rows), 1)

    def test_object_nodes_carry_centroid_extent(self):
        self.db.observe_object(
            description="red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        rec = self.db.all_records()[0]
        self.assertEqual(rec.spa.centroid, (1.0, 2.0, 0.5))
        self.assertEqual(rec.spa.extent, (0.5, 0.5, 1.0))

    def test_dedup_merges_temporal_keys(self):
        """Eq.5 dedup: same text + same place → merge, append timestamp."""
        self.db.observe_object(
            description="a red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=10.0,
        )
        ok = self.db.observe_object(
            description="a red chair",
            centroid=(1.0, 2.0, 0.5),
            extent=(0.5, 0.5, 1.0),
            timestamp=15.0,
        )
        self.assertFalse(ok)  # merged
        rec = self.db.all_records()[0]
        self.assertEqual(rec.tem.timestamps, [10.0, 15.0])

    def test_vector_storage_holds_embeddings(self):
        self.db.observe_object(
            description="red chair",
            centroid=(0, 0, 0),
            extent=(0.1, 0.1, 0.1),
            timestamp=0.0,
        )
        # The vector collection should have one entry
        rows = self.db.vector_storage.scroll("r4_objects")
        self.assertEqual(len(rows), 1)


class TestR4RetrievalViaStorage(unittest.TestCase):
    def setUp(self):
        self.db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
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

    def test_spatial_nearest_neighbour(self):
        ids = self.db._retrieve(k_spa_centroid=(1.1, 1.1, 0.5), k_spa_radius=1.0)
        self.assertEqual(len(ids), 1)
        rec = self.db.get_record(ids[0])
        self.assertIn("chair", rec.sem.description.lower())

    def test_temporal_window(self):
        ids = self.db._retrieve(k_t_min=15.0, k_t_max=25.0)
        self.assertEqual(len(ids), 1)
        rec = self.db.get_record(ids[0])
        self.assertIn("sofa", rec.sem.description.lower())

    def test_combined_spatial_temporal(self):
        ids = self.db._retrieve(
            k_spa_centroid=(0.9, 1.1, 0.5),
            k_spa_radius=1.0,
            k_t_min=5.0,
            k_t_max=15.0,
        )
        self.assertEqual(len(ids), 1)


class TestDirectModuleUsage(unittest.TestCase):
    """The DB can also be used as a plain unimem MemoryModule."""

    def test_write_via_memory_entry_uses_dedup_policy(self):
        db = R4KnowledgeDatabase(embedding_fn=MockEmbedding(dim=32))
        e1 = MemoryEntry(
            entry_id="obs1",
            text="a tall plant",
            spatial_keys=[(2.0, 3.0, 0.0, 0.4, 0.4, 1.5)],
            temporal_keys=[5.0],
        )
        ok = db.write(e1, MemoryContext())
        # Don't require a context — DedupPolicy ignores it.
        self.assertTrue(ok)
        rec = db.all_records()[0]
        self.assertEqual(rec.spa.centroid, (2.0, 3.0, 0.0))


if __name__ == "__main__":
    unittest.main()
