"""Tests for write / read / consolidation / forget policies."""
from __future__ import annotations

import unittest

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.policies.consolidation_policy import ConsolidationPolicy, Passthrough
from unimem.policies.forget_policy import ForgetPolicy, NoOp
from unimem.policies.read_policy import ConcatRead, FirstNonEmptyRead, ReadPolicy
from unimem.policies.write_policy import (
    AlwaysWrite,
    LambdaWritePolicy,
    NeverWrite,
    WritePolicy,
)


class _Stub(MemoryModule):
    """In-memory stub with optional consolidation."""

    def __init__(self, entries=None, consolidated=None):
        self._entries = list(entries) if entries else []
        self._consolidated = consolidated or []
        self.writes = 0

    def write(self, entry, context):
        self._entries.append(entry)
        self.writes += 1
        return True

    def read(self, query):
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()

    def stats(self):
        return {"count": len(self._entries)}

    def consolidate(self, other, context):
        return list(self._consolidated)


class TestWritePolicy(unittest.TestCase):
    def test_always_write(self):
        self.assertTrue(AlwaysWrite().should_write(_Stub(), MemoryEntry("e", "x"), MemoryContext()))

    def test_never_write(self):
        self.assertFalse(NeverWrite().should_write(_Stub(), MemoryEntry("e", "x"), MemoryContext()))

    def test_lambda_policy(self):
        # only admit entries whose text contains 'keep'
        pol = LambdaWritePolicy(lambda m, e, c: "keep" in e.text)
        self.assertTrue(pol.should_write(_Stub(), MemoryEntry("e", "keep this"), MemoryContext()))
        self.assertFalse(pol.should_write(_Stub(), MemoryEntry("e", "drop"), MemoryContext()))

    def test_lambda_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            LambdaWritePolicy("not callable")

    def test_write_policy_is_abc(self):
        with self.assertRaises(TypeError):
            WritePolicy()  # type: ignore[abstract]


class TestReadPolicy(unittest.TestCase):
    def test_concat_preserves_order_and_drops_scores(self):
        r1 = QueryResult(
            entries=[MemoryEntry("a", "x"), MemoryEntry("b", "y")],
            scores=[0.9, 0.8],
            source_node_id="n1",
        )
        r2 = QueryResult(entries=[MemoryEntry("c", "z")], source_node_id="n2")
        merged = ConcatRead().merge([r1, r2])
        self.assertEqual([e.entry_id for e in merged.entries], ["a", "b", "c"])
        self.assertIsNone(merged.scores)
        self.assertEqual(merged.metadata["per_source_counts"], {"n1": 2, "n2": 1})
        self.assertEqual(merged.metadata["n_sources"], 2)

    def test_concat_skips_empty(self):
        r1 = QueryResult()
        r2 = QueryResult(entries=[MemoryEntry("a", "x")], source_node_id="n2")
        merged = ConcatRead().merge([r1, r2])
        self.assertEqual(len(merged.entries), 1)

    def test_concat_empty_input(self):
        merged = ConcatRead().merge([])
        self.assertEqual(merged.entries, [])

    def test_first_non_empty(self):
        r1 = QueryResult()
        r2 = QueryResult(entries=[MemoryEntry("a", "x")], source_node_id="n2")
        r3 = QueryResult(entries=[MemoryEntry("b", "y")], source_node_id="n3")
        out = FirstNonEmptyRead().merge([r1, r2, r3])
        self.assertEqual(len(out.entries), 1)
        self.assertEqual(out.entries[0].entry_id, "a")
        self.assertEqual(out.source_node_id, "n2")

    def test_first_non_empty_all_empty(self):
        out = FirstNonEmptyRead().merge([QueryResult(), QueryResult()])
        self.assertEqual(out.entries, [])


class TestConsolidationPolicy(unittest.TestCase):
    def test_passthrough_delegates_to_source(self):
        to_extract = [MemoryEntry("fact1", "the chair is red")]
        source = _Stub(consolidated=to_extract)
        target = _Stub()
        out = Passthrough().extract(source, target, MemoryContext())
        self.assertEqual(out, to_extract)
        # And it's a copy, so caller can mutate safely
        out.append(MemoryEntry("extra", "x"))
        self.assertEqual(len(source._consolidated), 1)

    def test_consolidation_policy_is_abc(self):
        with self.assertRaises(TypeError):
            ConsolidationPolicy()  # type: ignore[abstract]


class TestForgetPolicy(unittest.TestCase):
    def test_noop(self):
        m = _Stub(entries=[MemoryEntry("a", "x")])
        n = NoOp().apply(m, MemoryContext())
        self.assertEqual(n, 0)
        self.assertEqual(m.count(), 1)

    def test_forget_policy_is_abc(self):
        with self.assertRaises(TypeError):
            ForgetPolicy()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
