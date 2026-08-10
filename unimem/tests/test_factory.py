"""Tests for the Registry and MemoryFactory."""
from __future__ import annotations

import unittest

from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.factory.memory_factory import MemoryFactory
from unimem.factory.registry import Registry
from unimem.policies.consolidation_policy import Passthrough
from unimem.policies.forget_policy import NoOp
from unimem.policies.read_policy import ConcatRead
from unimem.policies.write_policy import AlwaysWrite, NeverWrite


# --------------------------------------------------------------------------- #
# Minimal concrete module for registration
# --------------------------------------------------------------------------- #
class FakeEpisodic(MemoryModule):
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._entries = []

    def write(self, e, c):
        self._entries.append(e)
        return True

    def read(self, q):
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()

    def stats(self):
        return {"count": len(self._entries), "capacity": self.capacity}


class TestRegistryModules(unittest.TestCase):
    def test_register_and_create(self):
        r = Registry()
        r.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        m = r.create_module(MemorySlot.EM, "fake", capacity=42)
        self.assertIsInstance(m, FakeEpisodic)
        self.assertEqual(m.capacity, 42)

    def test_register_accepts_slot_string(self):
        r = Registry()
        r.register_module("episodic", "fake", FakeEpisodic)
        self.assertTrue(r.is_registered("EM", "fake"))

    def test_register_rejects_non_module(self):
        r = Registry()
        with self.assertRaises(TypeError):
            r.register_module(MemorySlot.EM, "bad", dict)  # type: ignore[arg-type]

    def test_register_rejects_instance(self):
        r = Registry()
        with self.assertRaises(TypeError):
            r.register_module(MemorySlot.EM, "bad", FakeEpisodic())  # type: ignore[arg-type]

    def test_create_unknown_raises_with_helpful_message(self):
        r = Registry()
        with self.assertRaises(KeyError) as cm:
            r.create_module(MemorySlot.EM, "ghost")
        self.assertIn("Available", str(cm.exception))

    def test_decorator_form(self):
        r = Registry()

        @r.register_module_decorator(MemorySlot.WM, "stub")
        class Stub(MemoryModule):
            def write(self, e, c): return True
            def read(self, q): return QueryResult()
            def clear(self): pass
            def stats(self): return {}

        self.assertTrue(r.is_registered(MemorySlot.WM, "stub"))
        m = r.create_module("WM", "stub")
        self.assertEqual(type(m).__name__, "Stub")

    def test_list_implementations_all_slots(self):
        r = Registry()
        r.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        r.register_module(MemorySlot.EM, "other", FakeEpisodic)
        r.register_module(MemorySlot.SM, "x", FakeEpisodic)
        listing = r.list_implementations()
        self.assertEqual(listing["EM"], ["fake", "other"])
        self.assertEqual(listing["SM"], ["x"])
        # Empty slots appear too
        self.assertEqual(listing["WM"], [])

    def test_list_implementations_single_slot(self):
        r = Registry()
        r.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        self.assertEqual(r.list_implementations(MemorySlot.EM), {"EM": ["fake"]})
        self.assertEqual(r.list_implementations("sm"), {"SM": []})

    def test_is_registered(self):
        r = Registry()
        r.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        self.assertTrue(r.is_registered(MemorySlot.EM, "fake"))
        self.assertFalse(r.is_registered(MemorySlot.EM, "ghost"))
        self.assertFalse(r.is_registered(MemorySlot.WM, "fake"))


class TestRegistryPolicies(unittest.TestCase):
    def test_register_and_create_policy(self):
        r = Registry()
        r.register_policy("write", "always", AlwaysWrite)
        r.register_policy("write", "never", NeverWrite)
        a = r.create_policy("write", "always")
        n = r.create_policy("write", "never")
        self.assertIsInstance(a, AlwaysWrite)
        self.assertIsInstance(n, NeverWrite)

    def test_register_all_policy_kinds(self):
        r = Registry()
        r.register_policy("write", "a", AlwaysWrite)
        r.register_policy("read", "a", ConcatRead)
        r.register_policy("consolidation", "a", Passthrough)
        r.register_policy("forget", "a", NoOp)
        listing = r.list_policies()
        for k in ("write", "read", "consolidation", "forget"):
            self.assertEqual(listing[k], ["a"])

    def test_register_rejects_wrong_type(self):
        r = Registry()
        # write policy slot given a read policy class
        with self.assertRaises(TypeError):
            r.register_policy("write", "bad", ConcatRead)

    def test_register_rejects_unknown_policy_type(self):
        r = Registry()
        with self.assertRaises(ValueError):
            r.register_policy("bogus", "x", AlwaysWrite)  # type: ignore[arg-type]

    def test_create_unknown_policy(self):
        r = Registry()
        with self.assertRaises(KeyError):
            r.create_policy("write", "ghost")

    def test_list_policies_filtered(self):
        r = Registry()
        r.register_policy("write", "a", AlwaysWrite)
        r.register_policy("read", "b", ConcatRead)
        self.assertEqual(r.list_policies("write"), {"write": ["a"]})


class TestMemoryFactory(unittest.TestCase):
    def test_factory_default_creates_empty_registry(self):
        f = MemoryFactory()
        self.assertEqual(sum(len(v) for v in f.registry.list_implementations().values()), 0)

    def test_factory_register_and_make(self):
        f = MemoryFactory()
        f.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        m = f.make_module("EM", "fake", capacity=5)
        self.assertEqual(m.capacity, 5)

    def test_factory_register_returns_self_for_chaining(self):
        f = MemoryFactory()
        ret = f.register_module(MemorySlot.EM, "fake", FakeEpisodic)
        self.assertIs(ret, f)

    def test_factory_decorator_passthrough(self):
        f = MemoryFactory()

        @f.register_module_decorator(MemorySlot.PM, "x")
        class Skill(MemoryModule):
            def write(self, e, c): return True
            def read(self, q): return QueryResult()
            def clear(self): pass
            def stats(self): return {}

        self.assertTrue(f.registry.is_registered(MemorySlot.PM, "x"))


if __name__ == "__main__":
    unittest.main()
