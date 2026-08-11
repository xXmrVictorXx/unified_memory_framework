"""Tests for the VectorStorage ABC + InMemoryVectorStorage backend."""
from __future__ import annotations

import math
import unittest

from unimem.vector_storage import (
    InMemoryVectorStorage,
    VectorStorage,
    create_vector_storage,
)


class TestVectorStorageFactory(unittest.TestCase):
    def test_default_is_in_memory(self):
        vs = create_vector_storage({})
        self.assertIsInstance(vs, InMemoryVectorStorage)

    def test_explicit_memory_backend(self):
        vs = create_vector_storage({"backend": "memory"})
        self.assertIsInstance(vs, InMemoryVectorStorage)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_vector_storage({"backend": "milvus"})


class TestInMemoryCollection(unittest.TestCase):
    def setUp(self):
        self.vs = InMemoryVectorStorage()
        self.vs.create_collection("objects", vector_dim=3)

    def test_create_collection_is_idempotent(self):
        self.assertTrue(self.vs.create_collection("objects", vector_dim=3))
        self.assertTrue(self.vs.collection_exists("objects"))

    def test_invalid_distance_metric(self):
        with self.assertRaises(ValueError):
            self.vs.create_collection("bad", vector_dim=3, distance_metric="manhattan")

    def test_delete_collection(self):
        self.assertTrue(self.vs.delete_collection("objects"))
        self.assertFalse(self.vs.collection_exists("objects"))
        self.assertFalse(self.vs.delete_collection("objects"))

    def test_upsert_and_get(self):
        self.vs.upsert("objects", "p1", [1.0, 0.0, 0.0], {"label": "red"})
        vec, payload = self.vs.get("objects", "p1")
        self.assertEqual(vec, [1.0, 0.0, 0.0])
        self.assertEqual(payload, {"label": "red"})

    def test_upsert_replaces_existing(self):
        self.vs.upsert("objects", "p1", [1.0, 0.0, 0.0], {"v": 1})
        self.vs.upsert("objects", "p1", [0.0, 1.0, 0.0], {"v": 2})
        vec, payload = self.vs.get("objects", "p1")
        self.assertEqual(vec, [0.0, 1.0, 0.0])
        self.assertEqual(payload, {"v": 2})
        self.assertEqual(self.vs.count("objects"), 1)

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.vs.upsert("objects", "p1", [1.0, 0.0], {})

    def test_get_missing_point(self):
        self.assertIsNone(self.vs.get("objects", "nope"))

    def test_get_missing_collection(self):
        self.assertIsNone(self.vs.get("nope", "p1"))

    def test_upsert_missing_collection_raises(self):
        with self.assertRaises(KeyError):
            self.vs.upsert("nope", "p1", [1, 2, 3])

    def test_delete(self):
        self.vs.upsert("objects", "p1", [1, 0, 0])
        self.assertTrue(self.vs.delete("objects", "p1"))
        self.assertFalse(self.vs.delete("objects", "p1"))
        self.assertEqual(self.vs.count("objects"), 0)

    def test_count(self):
        self.assertEqual(self.vs.count("objects"), 0)
        for i in range(5):
            self.vs.upsert("objects", f"p{i}", [float(i), 0, 0])
        self.assertEqual(self.vs.count("objects"), 5)


class TestInMemorySearch(unittest.TestCase):
    def setUp(self):
        self.vs = InMemoryVectorStorage()
        self.vs.create_collection("docs", vector_dim=3, distance_metric="cosine")
        # Three orthogonal-ish vectors
        self.vs.upsert("docs", "a", [1.0, 0.0, 0.0], {"group": "x"})
        self.vs.upsert("docs", "b", [0.0, 1.0, 0.0], {"group": "x"})
        self.vs.upsert("docs", "c", [0.0, 0.0, 1.0], {"group": "y"})

    def test_cosine_returns_best_match_first(self):
        results = self.vs.search("docs", [1.0, 0.1, 0.0], top_k=3)
        self.assertEqual(results[0][0], "a")
        self.assertGreater(results[0][1], results[1][1])

    def test_top_k_truncation(self):
        results = self.vs.search("docs", [1, 1, 1], top_k=2)
        self.assertEqual(len(results), 2)

    def test_score_threshold_filters(self):
        # cosine threshold 0.99 — only near-orthogonal matches kept
        results = self.vs.search(
            "docs", [1.0, 0.0, 0.0], top_k=10, score_threshold=0.99
        )
        self.assertEqual([r[0] for r in results], ["a"])

    def test_filter_conditions(self):
        results = self.vs.search(
            "docs", [1.0, 0.0, 0.0], top_k=10, filter_conditions={"group": "x"}
        )
        ids = sorted(r[0] for r in results)
        self.assertEqual(ids, ["a", "b"])

    def test_l2_metric(self):
        self.vs.create_collection("l2col", vector_dim=2, distance_metric="l2")
        self.vs.upsert("l2col", "a", [0.0, 0.0])
        self.vs.upsert("l2col", "b", [3.0, 4.0])
        results = self.vs.search("l2col", [0.0, 0.0], top_k=2)
        self.assertEqual(results[0][0], "a")
        self.assertEqual(results[1][0], "b")

    def test_scroll(self):
        rows = self.vs.scroll("docs")
        self.assertEqual(len(rows), 3)
        rows = self.vs.scroll("docs", filter_conditions={"group": "y"})
        self.assertEqual([r[0] for r in rows], ["c"])

    def test_scroll_limit(self):
        rows = self.vs.scroll("docs", limit=2)
        self.assertEqual(len(rows), 2)


class TestVectorStorageABC(unittest.TestCase):
    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            VectorStorage()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
