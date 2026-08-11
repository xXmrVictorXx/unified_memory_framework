"""Tests for the TimeIndex node helpers."""
from __future__ import annotations

import unittest

from unimem.core.entry import MemoryEntry
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage
from unimem.graph_storage.time_index import (
    attach_clip,
    attach_period,
    attach_timestamp,
    attach_timescale_bucket,
    write_with_time_index,
)


class TestTimeIndexHelpers(unittest.TestCase):
    def setUp(self):
        self.gs = InMemoryGraphStorage()
        self.entry = MemoryEntry(
            entry_id="e1", text="person walking", semantic_keys=["person"]
        )
        self.gs.add_memory_node(MemorySlot.EM, self.entry)

    def _attached_time_ids(self, memory_id):
        neighbours = self.gs.get_neighbors(memory_id, "AT_TIME", "out")
        return [n[0] for n in neighbours]

    def test_attach_timestamp(self):
        tid = attach_timestamp(self.gs, "e1", 5.5)
        self.assertTrue(tid.startswith("time-e1-"))
        node = self.gs.get_node(tid)
        self.assertIn("TimeIndex", node["labels"])
        self.assertEqual(node["properties"]["timestamp"], 5.5)
        self.assertIn(tid, self._attached_time_ids("e1"))

    def test_attach_clip(self):
        tid = attach_clip(self.gs, "e1", 3, start_t=10.0, end_t=15.0)
        node = self.gs.get_node(tid)
        self.assertEqual(node["properties"]["clip_index"], 3)
        self.assertEqual(node["properties"]["start_t"], 10.0)
        self.assertEqual(node["properties"]["end_t"], 15.0)

    def test_attach_period(self):
        tid = attach_period(self.gs, "e1", "00:00:10-00:00:30", start_t=10.0)
        node = self.gs.get_node(tid)
        self.assertEqual(node["properties"]["period"], "00:00:10-00:00:30")
        self.assertEqual(node["properties"]["start_t"], 10.0)

    def test_attach_timescale_bucket(self):
        tid = attach_timescale_bucket(self.gs, "e1", 30.0)
        node = self.gs.get_node(tid)
        self.assertEqual(node["properties"]["timescale"], 30.0)
        self.assertEqual(node["properties"]["kind"], "bucket")

    def test_write_with_time_index_clip(self):
        entry = MemoryEntry(entry_id="e2", text="caption")
        nid = write_with_time_index(
            self.gs, MemorySlot.EM, entry, clip_index=0, start_t=0.0, end_t=1.0
        )
        self.assertEqual(nid, "e2")
        node = self.gs.get_node("e2")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.EM.value, node["labels"])
        # Should have an AT_TIME edge to a TimeIndex
        self.assertEqual(len(self._attached_time_ids("e2")), 1)

    def test_write_with_time_index_period(self):
        entry = MemoryEntry(entry_id="e3", text="walking")
        write_with_time_index(
            self.gs, MemorySlot.EM, entry, period="00:00:00-00:00:30"
        )
        time_ids = self._attached_time_ids("e3")
        self.assertEqual(len(time_ids), 1)
        node = self.gs.get_node(time_ids[0])
        self.assertEqual(node["properties"]["period"], "00:00:00-00:00:30")

    def test_deterministic_time_index_id(self):
        # Same props => same id
        tid1 = attach_timestamp(self.gs, "e1", 5.0)
        # Attach again — should overwrite (same id, MERGE node)
        tid2 = attach_timestamp(self.gs, "e1", 5.0)
        self.assertEqual(tid1, tid2)
        # Only one AT_TIME edge for the same time props
        matching = [
            n
            for n in self._attached_time_ids("e1")
            if n == tid1
        ]
        self.assertEqual(len(matching), 1)


if __name__ == "__main__":
    unittest.main()
