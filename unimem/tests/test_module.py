"""Tests verifying ABC compliance and that minimal concrete modules work."""
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


class TestABCInheritance(unittest.TestCase):
    def test_module_is_abc(self):
        self.assertTrue(issubclass(MemoryModule, ABC))
        self.assertTrue(hasattr(MemoryModule, "__abstractmethods__"))
        # The 4 abstract methods named in the plan
        self.assertEqual(
            MemoryModule.__abstractmethods__,
            {"write", "read", "clear", "stats"},
        )

    def test_slot_abcs_extend_module(self):
        for slot_cls in (
            WorkingMemoryABC,
            SceneGraphMemoryABC,
            SpatialGeometricMemoryABC,
            EpisodicMemoryABC,
            SemanticMemoryABC,
            ProceduralMemoryABC,
        ):
            self.assertTrue(issubclass(slot_cls, MemoryModule), slot_cls)

    def test_slot_abcs_add_methods(self):
        # Each slot ABC should expose its slot-specific methods.
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

    def test_cannot_instantiate_module_without_methods(self):
        with self.assertRaises(TypeError):
            MemoryModule()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# A minimal concrete implementation used by graph/factory tests later.
# --------------------------------------------------------------------------- #
class _StampMemory(MemoryModule):
    """Records every call so we can inspect graph behaviour."""

    def __init__(self, name: str = "stub"):
        self.name = name
        self._entries = []
        self.written: list = []
        self.reads: list = []
        self.updates = 0

    def write(self, entry, context):
        self.written.append(entry.entry_id)
        self._entries.append(entry)
        return True

    def read(self, query):
        self.reads.append(query)
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()

    def stats(self):
        return {"count": len(self._entries), "name": self.name}

    def update(self, context):
        self.updates += 1


class TestMinimalConcreteModule(unittest.TestCase):
    def test_write_read_clear_stats(self):
        m = _StampMemory()
        ctx = MemoryContext()
        e = MemoryEntry("e1", "hello")
        self.assertTrue(m.write(e, ctx))
        result = m.read(Query())
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].entry_id, "e1")
        self.assertEqual(m.stats()["count"], 1)
        m.clear()
        self.assertEqual(m.read(Query()).entries, [])

    def test_count_helper_reads_stats(self):
        m = _StampMemory()
        m.write(MemoryEntry("a", "x"), MemoryContext())
        m.write(MemoryEntry("b", "y"), MemoryContext())
        self.assertEqual(m.count(), 2)

    def test_default_update_and_consolidate_are_safe(self):
        m = _StampMemory()
        # Default update is overridden here to count, but base class default is no-op
        other = _StampMemory()
        self.assertEqual(other.consolidate(m, MemoryContext()), [])
        # Update should not raise.
        MemoryModule.update(other, MemoryContext())

    def test_default_count_returns_minus_one_when_stats_unknown(self):
        class NoStat(MemoryModule):
            def write(self, e, c): return True
            def read(self, q): return QueryResult()
            def clear(self): pass
            def stats(self): return {"weird_key": 7}

        self.assertEqual(NoStat().count(), -1)


if __name__ == "__main__":
    unittest.main()
