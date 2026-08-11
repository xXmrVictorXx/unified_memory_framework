"""Tests verifying the storage-aware MemoryModule + slot ABC mixins."""
from __future__ import annotations

import unittest
from abc import ABC

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryBuilder, QueryResult
from unimem.core.slot_abc import (
    EpisodicMemoryABC,
    ProceduralMemoryABC,
    SceneGraphMemoryABC,
    SemanticMemoryABC,
    SpatialGeometricMemoryABC,
    WorkingMemoryABC,
)
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage


class TestSlotABCInheritance(unittest.TestCase):
    def test_module_not_abc_itself(self):
        # MemoryModule now has storage-backed defaults — no longer abstract.
        self.assertFalse(issubclass(MemoryModule, ABC))
        m = MemoryModule(slot=MemorySlot.WM)
        self.assertIsInstance(m, MemoryModule)

    def test_slot_abcs_extend_module_and_abc(self):
        for slot_cls in (
            WorkingMemoryABC,
            SceneGraphMemoryABC,
            SpatialGeometricMemoryABC,
            EpisodicMemoryABC,
            SemanticMemoryABC,
            ProceduralMemoryABC,
        ):
            self.assertTrue(issubclass(slot_cls, MemoryModule), slot_cls)
            self.assertTrue(issubclass(slot_cls, ABC), slot_cls)

    def test_slot_abcs_declare_abstract_methods(self):
        # The slot ABCs still enforce their slot-specific methods via ABC.
        self.assertIn("get_current", WorkingMemoryABC.__abstractmethods__)
        self.assertIn("set_current", WorkingMemoryABC.__abstractmethods__)
        self.assertIn("add_object", SceneGraphMemoryABC.__abstractmethods__)
        self.assertIn("get_children", SceneGraphMemoryABC.__abstractmethods__)
        self.assertIn("get_object_by_id", SceneGraphMemoryABC.__abstractmethods__)
        self.assertIn("is_navigable", SpatialGeometricMemoryABC.__abstractmethods__)
        self.assertIn("get_region", SpatialGeometricMemoryABC.__abstractmethods__)
        self.assertIn("append_event", EpisodicMemoryABC.__abstractmethods__)
        self.assertIn("get_timeline", EpisodicMemoryABC.__abstractmethods__)
        self.assertIn("add_fact", SemanticMemoryABC.__abstractmethods__)
        self.assertIn("query_facts", SemanticMemoryABC.__abstractmethods__)
        self.assertIn("add_skill", ProceduralMemoryABC.__abstractmethods__)
        self.assertIn("find_skill", ProceduralMemoryABC.__abstractmethods__)

    def test_cannot_instantiate_slot_abc_without_methods(self):
        with self.assertRaises(TypeError):
            WorkingMemoryABC(slot=MemorySlot.WM)  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# A minimal storage-backed module — the recommended modern pattern.
# --------------------------------------------------------------------------- #
class TestStorageBackedDefaults(unittest.TestCase):
    def setUp(self):
        self.gs = InMemoryGraphStorage()
        self.m = MemoryModule(slot=MemorySlot.EM, graph_storage=self.gs)

    def test_write_routes_through_storage(self):
        entry = MemoryEntry("e1", "hello", semantic_keys=["greeting"])
        ctx = MemoryContext()
        self.assertTrue(self.m.write(entry, ctx))
        # Node should be in the storage
        node = self.gs.get_node("e1")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.EM.value, node["labels"])

    def test_read_returns_storage_backed_entries(self):
        self.m.write(MemoryEntry("e1", "hello", semantic_keys=["x"]), MemoryContext())
        self.m.write(MemoryEntry("e2", "world", semantic_keys=["y"]), MemoryContext())
        result = self.m.read(Query())
        self.assertEqual(len(result.entries), 2)
        self.assertEqual(result.source_slot, MemorySlot.EM.value)

    def test_read_filter_by_semantic_keys(self):
        self.m.write(MemoryEntry("e1", "alpha", semantic_keys=["x"]), MemoryContext())
        self.m.write(MemoryEntry("e2", "beta", semantic_keys=["y"]), MemoryContext())
        result = self.m.read(QueryBuilder().with_semantic("x").build())
        self.assertEqual([e.entry_id for e in result.entries], ["e1"])

    def test_clear_deletes_all_slot_nodes(self):
        self.m.write(MemoryEntry("e1", "a"), MemoryContext())
        self.m.write(MemoryEntry("e2", "b"), MemoryContext())
        self.m.clear()
        self.assertEqual(self.m.count(), 0)

    def test_stats_reflect_storage_count(self):
        self.m.write(MemoryEntry("e1", "a"), MemoryContext())
        self.m.write(MemoryEntry("e2", "b"), MemoryContext())
        s = self.m.stats()
        self.assertEqual(s["count"], 2)
        self.assertEqual(s["slot"], MemorySlot.EM.value)

    def test_count_helper_reads_stats(self):
        self.m.write(MemoryEntry("a", "x"), MemoryContext())
        self.m.write(MemoryEntry("b", "y"), MemoryContext())
        self.assertEqual(self.m.count(), 2)


class TestStorageLessModuleRaises(unittest.TestCase):
    def test_write_without_storage_raises(self):
        m = MemoryModule(slot=MemorySlot.WM)
        with self.assertRaises(RuntimeError):
            m.write(MemoryEntry("a", "x"), MemoryContext())

    def test_read_without_storage_raises(self):
        m = MemoryModule(slot=MemorySlot.WM)
        with self.assertRaises(RuntimeError):
            m.read(Query())

    def test_clear_without_storage_raises(self):
        m = MemoryModule(slot=MemorySlot.WM)
        with self.assertRaises(RuntimeError):
            m.clear()

    def test_stats_without_storage_returns_zero(self):
        # stats is non-mutating — should not raise; returns 0 count.
        m = MemoryModule(slot=MemorySlot.WM)
        s = m.stats()
        self.assertEqual(s["count"], 0)


class TestLegacyOverrides(unittest.TestCase):
    """Subclasses can still override write/read for custom in-memory behaviour."""

    def test_override_bypasses_storage(self):
        class StubMemory(MemoryModule):
            def __init__(self):
                super().__init__(slot=MemorySlot.WM)
                self._entries = []
                self.written = []

            def write(self, entry, ctx):
                self.written.append(entry.entry_id)
                self._entries.append(entry)
                return True

            def read(self, query):
                return QueryResult(entries=list(self._entries))

            def clear(self):
                self._entries.clear()

            def stats(self):
                return {"count": len(self._entries)}

        m = StubMemory()
        ctx = MemoryContext()
        e = MemoryEntry("e1", "hello")
        self.assertTrue(m.write(e, ctx))
        result = m.read(Query())
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(m.stats()["count"], 1)
        m.clear()
        self.assertEqual(m.read(Query()).entries, [])


if __name__ == "__main__":
    unittest.main()
