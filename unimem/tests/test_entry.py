"""Tests for MemoryEntry + MultiAxisIndex."""
from __future__ import annotations

import unittest

from unimem.core.entry import MemoryEntry, MultiAxisIndex


class TestMemoryEntry(unittest.TestCase):
    def test_defaults_are_empty_collections(self):
        e = MemoryEntry("e1", "a red chair")
        self.assertEqual(e.semantic_keys, [])
        self.assertEqual(e.spatial_keys, [])
        self.assertEqual(e.temporal_keys, [])
        self.assertEqual(e.metadata, {})
        self.assertIsNone(e.payload)
        self.assertIsNone(e.source_slot)

    def test_spatial_keys_normalised_to_float_tuples(self):
        e = MemoryEntry("e1", "x", spatial_keys=[(1, 2), (3.0, 4.0)])
        self.assertEqual(e.spatial_keys, [(1.0, 2.0), (3.0, 4.0)])
        for k in e.spatial_keys:
            self.assertIsInstance(k, tuple)

    def test_temporal_keys_coerced_to_float(self):
        e = MemoryEntry("e1", "x", temporal_keys=[1, 2.5, "3"])
        self.assertEqual(e.temporal_keys, [1.0, 2.5, 3.0])

    def test_metadata_is_copied(self):
        md = {"conf": 0.9}
        e = MemoryEntry("e1", "x", metadata=md)
        md["conf"] = 0.1
        self.assertEqual(e.metadata["conf"], 0.9)

    def test_matches_predicates(self):
        e = MemoryEntry(
            "e1",
            "red chair near window",
            semantic_keys=["chair", "red"],
            spatial_keys=[(1.2, 3.4)],
            temporal_keys=[1234.5],
        )
        self.assertTrue(e.matches_semantic("chair"))
        self.assertFalse(e.matches_semantic("sofa"))
        self.assertTrue(e.matches_spatial((1.2, 3.4)))
        self.assertTrue(e.matches_temporal(1234.5))

    def test_add_helpers_return_self_for_chaining(self):
        e = MemoryEntry("e1", "x")
        ret = e.add_semantic("a", "b").add_spatial((1, 2)).add_temporal(1.0, 2.0)
        self.assertIs(ret, e)
        self.assertEqual(e.semantic_keys, ["a", "b"])
        self.assertEqual(e.spatial_keys, [(1.0, 2.0)])
        self.assertEqual(e.temporal_keys, [1.0, 2.0])

    def test_equality_by_id(self):
        a = MemoryEntry("e1", "x")
        b = MemoryEntry("e1", "completely different text")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class TestMultiAxisIndex(unittest.TestCase):
    def _make_entries(self):
        return [
            MemoryEntry(
                "e1",
                "red chair",
                semantic_keys=["chair", "red", "furniture"],
                spatial_keys=[(1.0, 2.0)],
                temporal_keys=[10.0],
            ),
            MemoryEntry(
                "e2",
                "blue sofa",
                semantic_keys=["sofa", "blue", "furniture"],
                spatial_keys=[(3.0, 4.0)],
                temporal_keys=[20.0],
            ),
            MemoryEntry(
                "e3",
                "red sofa",
                semantic_keys=["sofa", "red"],
                spatial_keys=[(1.0, 2.0)],
                temporal_keys=[15.0],
            ),
        ]

    def test_add_and_size(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        self.assertEqual(len(idx), 3)

    def test_semantic_lookup_single_key(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        self.assertEqual(idx.lookup_semantic("chair"), {"e1"})
        self.assertEqual(idx.lookup_semantic("sofa"), {"e2", "e3"})
        self.assertEqual(idx.lookup_semantic("missing"), set())

    def test_spatial_lookup(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        self.assertEqual(idx.lookup_spatial((1.0, 2.0)), {"e1", "e3"})
        self.assertEqual(idx.lookup_spatial((9.0, 9.0)), set())

    def test_temporal_range_lookup(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        self.assertEqual(idx.lookup_temporal(0, 100), {"e1", "e2", "e3"})
        self.assertEqual(idx.lookup_temporal(12, 18), {"e3"})
        self.assertEqual(idx.lookup_temporal(10, 10), {"e1"})
        self.assertEqual(idx.lookup_temporal(100, 200), set())
        # Unbounded
        self.assertEqual(idx.lookup_temporal(), {"e1", "e2", "e3"})

    def test_multi_axis_intersection(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        # Red AND at (1,2) -> e1, e3 ; (red AND sofa) -> e3
        self.assertEqual(
            idx.lookup(semantic=["red"], spatial=[(1.0, 2.0)]), {"e1", "e3"}
        )
        # Default within-axis = intersect: red AND sofa -> e3
        self.assertEqual(idx.lookup(semantic=["red", "sofa"]), {"e3"})
        # Temporal window constrains
        self.assertEqual(
            idx.lookup(semantic=["sofa"], t_min=0, t_max=17), {"e3"}
        )

    def test_multi_axis_intersect_within_axis_default(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        # chair AND blue -> no entry has both
        self.assertEqual(idx.lookup(semantic=["chair", "blue"]), set())

    def test_multi_axis_union_within_axis_explicit(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        # chair OR blue -> e1 (chair) + e2 (blue)
        self.assertEqual(
            idx.lookup(semantic=["chair", "blue"], within_axis_op="union"),
            {"e1", "e2"},
        )
        # red OR sofa -> all three
        self.assertEqual(
            idx.lookup(semantic=["red", "sofa"], within_axis_op="union"),
            {"e1", "e2", "e3"},
        )

    def test_lookup_rejects_bad_within_axis_op(self):
        idx = MultiAxisIndex()
        with self.assertRaises(ValueError):
            idx.lookup(semantic=["a"], within_axis_op="bogus")

    def test_lookup_with_no_constraints_returns_all(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        self.assertEqual(len(idx.lookup()), 3)
        # require_any_axis forces empty when nothing was supplied
        self.assertEqual(idx.lookup(require_any_axis=True), set())

    def test_remove_drops_everywhere(self):
        idx = MultiAxisIndex()
        entries = self._make_entries()
        for e in entries:
            idx.add(e)
        idx.remove("e1")
        self.assertNotIn("e1", idx.lookup_semantic("chair"))
        self.assertNotIn("e1", idx.lookup_spatial((1.0, 2.0)))
        self.assertNotIn("e1", idx.lookup_temporal(0, 100))
        self.assertEqual(len(idx), 2)

    def test_re_add_is_idempotent(self):
        idx = MultiAxisIndex()
        e = self._make_entries()[0]
        idx.add(e)
        idx.add(e)
        self.assertEqual(len(idx), 1)

    def test_clear(self):
        idx = MultiAxisIndex()
        for e in self._make_entries():
            idx.add(e)
        idx.clear()
        self.assertEqual(len(idx), 0)
        self.assertEqual(idx.lookup_semantic("chair"), set())

    def test_remove_unknown_id_is_safe(self):
        idx = MultiAxisIndex()
        idx.add(self._make_entries()[0])
        idx.remove("nonexistent")
        self.assertEqual(len(idx), 1)


if __name__ == "__main__":
    unittest.main()
