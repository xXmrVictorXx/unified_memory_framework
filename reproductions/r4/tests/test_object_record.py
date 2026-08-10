"""Tests for R4 ObjectRecord data structures."""
from __future__ import annotations

import unittest

from reproductions.r4.memory.object_record import (
    ObjectRecord,
    SemanticAxis,
    SpatialAxis,
    TemporalAxis,
)


class TestAxes(unittest.TestCase):
    def test_semantic_axis_normalises_embedding(self):
        s = SemanticAxis("a chair", embedding=(0.1, 0.2, 0.3))
        self.assertEqual(s.embedding, [0.1, 0.2, 0.3])
        self.assertIsInstance(s.embedding, list)

    def test_spatial_axis_coerces_to_float(self):
        spa = SpatialAxis(centroid=(1, 2, 3), extent=("0.1", "0.2", "0.3"))
        self.assertEqual(spa.centroid, (1.0, 2.0, 3.0))
        self.assertEqual(spa.extent, (0.1, 0.2, 0.3))

    def test_spatial_axis_rejects_non_3d(self):
        with self.assertRaises(ValueError):
            SpatialAxis(centroid=(1, 2), extent=(1, 2, 3))

    def test_temporal_axis_observe_keeps_sorted(self):
        t = TemporalAxis()
        t.observe(3.0)
        t.observe(1.0)
        t.observe(2.0)
        self.assertEqual(t.timestamps, [1.0, 2.0, 3.0])
        self.assertEqual(t.first_seen, 1.0)
        self.assertEqual(t.last_seen, 3.0)

    def test_temporal_axis_dedup(self):
        t = TemporalAxis()
        t.observe(1.0)
        t.observe(1.0)
        self.assertEqual(t.timestamps, [1.0])

    def test_temporal_axis_overlaps(self):
        t = TemporalAxis(timestamps=[1.0, 5.0, 10.0])
        self.assertTrue(t.overlaps(0, 2))
        self.assertTrue(t.overlaps(5, 5))
        self.assertFalse(t.overlaps(6, 9))
        self.assertTrue(t.overlaps(None, None))
        self.assertFalse(TemporalAxis().overlaps(0, 10))


class TestObjectRecord(unittest.TestCase):
    def test_to_text_roundtrip(self):
        r = ObjectRecord(
            unique_id="obj-0001",
            sem=SemanticAxis("a red chair"),
            spa=SpatialAxis(centroid=(1.5, 2.0, 0.5), extent=(0.6, 0.6, 1.0)),
            tem=TemporalAxis(timestamps=[1.0, 5.0]),
        )
        text = r.to_text()
        self.assertIn("obj-0001", text)
        self.assertIn("red chair", text)
        self.assertIn("1.50", text)


if __name__ == "__main__":
    unittest.main()
