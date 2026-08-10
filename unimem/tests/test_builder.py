"""Tests for the declarative MemoryGraphBuilder."""
from __future__ import annotations

import unittest

from unimem.core.module import MemoryModule
from unimem.core.query import QueryResult
from unimem.core.slots import MemorySlot
from unimem.factory.registry import Registry
from unimem.graph.builder import EdgeSpec, GraphSpec, MemoryGraphBuilder, NodeSpec
from unimem.graph.edge import EdgeKind
from unimem.graph.graph import MemoryGraph
from unimem.policies.consolidation_policy import Passthrough
from unimem.policies.forget_policy import NoOp
from unimem.policies.read_policy import ConcatRead
from unimem.policies.write_policy import AlwaysWrite, NeverWrite


# --------------------------------------------------------------------------- #
# Minimal modules / helpers
# --------------------------------------------------------------------------- #
class StubMod(MemoryModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._entries = []

    def write(self, e, c):
        self._entries.append(e)
        return True

    def read(self, q):
        return QueryResult(entries=list(self._entries))

    def clear(self):
        self._entries.clear()

    def stats(self):
        return {"count": len(self._entries)}


def _registry_with_stubs() -> Registry:
    r = Registry()
    for slot in MemorySlot:
        r.register_module(slot, "stub", StubMod)
    r.register_policy("write", "always", AlwaysWrite)
    r.register_policy("write", "never", NeverWrite)
    r.register_policy("read", "concat", ConcatRead)
    r.register_policy("consolidation", "passthrough", Passthrough)
    r.register_policy("forget", "noop", NoOp)
    return r


# --------------------------------------------------------------------------- #
# Spec dataclass behaviour
# --------------------------------------------------------------------------- #
class TestSpecs(unittest.TestCase):
    def test_node_spec_defaults(self):
        ns = NodeSpec(node_id="wm", slot=MemorySlot.WM, impl="stub")
        self.assertEqual(ns.kwargs, {})
        self.assertEqual(ns.metadata, {})

    def test_edge_spec_defaults(self):
        es = EdgeSpec(source="a", target="b", kind="feeds")
        self.assertIsNone(es.policy_type)
        self.assertEqual(es.policy_kwargs, {})

    def test_graph_spec_from_dict(self):
        d = {
            "nodes": [
                {"node_id": "wm", "slot": "WM", "impl": "stub"},
                {"node_id": "em", "slot": "episodic", "impl": "stub"},
            ],
            "edges": [
                {"source": "wm", "target": "em", "kind": "feeds"}
            ],
            "default_write_policy": {"name": "always"},
        }
        spec = GraphSpec.from_dict(d)
        self.assertEqual(len(spec.nodes), 2)
        self.assertEqual(spec.nodes[0].node_id, "wm")
        self.assertEqual(len(spec.edges), 1)
        self.assertEqual(spec.default_write_policy, {"name": "always"})

    def test_graph_spec_to_dict_roundtrip(self):
        spec = GraphSpec(
            nodes=[NodeSpec("wm", MemorySlot.WM, "stub")],
            edges=[EdgeSpec("wm", "em", EdgeKind.FEEDS)],
        )
        d = spec.to_dict()
        self.assertEqual(d["nodes"][0]["slot"], "working_memory")
        self.assertEqual(d["edges"][0]["kind"], "feeds")
        # And rebuild
        spec2 = GraphSpec.from_dict(d)
        self.assertEqual(len(spec2.nodes), 1)
        self.assertEqual(spec2.nodes[0].slot, MemorySlot.WM)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class TestBuilder(unittest.TestCase):
    def test_build_minimal_graph(self):
        b = MemoryGraphBuilder(_registry_with_stubs())
        g = b.build(
            GraphSpec(
                nodes=[
                    NodeSpec("wm", MemorySlot.WM, "stub"),
                    NodeSpec("em", MemorySlot.EM, "stub"),
                ],
                edges=[EdgeSpec("wm", "em", EdgeKind.FEEDS)],
            )
        )
        self.assertIsInstance(g, MemoryGraph)
        self.assertEqual(len(g), 2)
        self.assertEqual(len(g.edges()), 1)

    def test_build_from_dict(self):
        b = MemoryGraphBuilder(_registry_with_stubs())
        g = b.build(
            {
                "nodes": [
                    {"node_id": "wm", "slot": "WM", "impl": "stub"},
                    {"node_id": "em", "slot": "EM", "impl": "stub"},
                ],
                "edges": [{"source": "wm", "target": "em", "kind": "feeds"}],
            }
        )
        self.assertEqual(len(g), 2)

    def test_build_rejects_bad_spec_type(self):
        b = MemoryGraphBuilder(_registry_with_stubs())
        with self.assertRaises(TypeError):
            b.build(42)  # type: ignore[arg-type]

    def test_build_passes_module_kwargs(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        g = b.build(
            GraphSpec(
                nodes=[NodeSpec("wm", MemorySlot.WM, "stub", kwargs={"cap": 7})]
            )
        )
        mod = g.get_node("wm").module
        self.assertEqual(mod.kwargs, {"cap": 7})

    def test_build_attaches_edge_policy_by_name(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        g = b.build(
            GraphSpec(
                nodes=[
                    NodeSpec("wm", MemorySlot.WM, "stub"),
                    NodeSpec("sg", MemorySlot.SG, "stub"),
                ],
                edges=[
                    EdgeSpec(
                        "wm", "sg", EdgeKind.FEEDS,
                        policy_type="write", policy_name="never",
                    )
                ],
            )
        )
        edge = g.edges_of("wm", EdgeKind.FEEDS)[0]
        self.assertIsInstance(edge.policy, NeverWrite)

    def test_build_edge_policy_with_kwargs(self):
        r = _registry_with_stubs()
        # Custom policy that takes a kwarg
        from unimem.policies.write_policy import LambdaWritePolicy

        r.register_policy("write", "lambda", LambdaWritePolicy)
        b = MemoryGraphBuilder(r)
        g = b.build(
            GraphSpec(
                nodes=[
                    NodeSpec("a", MemorySlot.WM, "stub"),
                    NodeSpec("b", MemorySlot.WM, "stub"),
                ],
                edges=[
                    EdgeSpec(
                        "a", "b", EdgeKind.FEEDS,
                        policy_type="write", policy_name="lambda",
                        policy_kwargs={"fn": lambda m, e, c: False},
                    )
                ],
            )
        )
        edge = g.edges_of("a", EdgeKind.FEEDS)[0]
        self.assertIsInstance(edge.policy, LambdaWritePolicy)

    def test_build_default_policies_resolved(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        g = b.build(
            GraphSpec(
                nodes=[NodeSpec("wm", MemorySlot.WM, "stub")],
                default_write_policy={"name": "always"},
                default_read_policy={"name": "concat"},
                default_forget_policy={"name": "noop"},
            )
        )
        self.assertIsInstance(g.default_write_policy, AlwaysWrite)
        self.assertIsInstance(g.default_read_policy, ConcatRead)
        self.assertIsInstance(g.default_forget_policy, NoOp)

    def test_build_default_policy_missing_name_raises(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        with self.assertRaises(ValueError):
            b.build(
                GraphSpec(
                    nodes=[NodeSpec("wm", MemorySlot.WM, "stub")],
                    default_write_policy={"kwargs": {}},
                )
            )

    def test_build_default_policy_bad_type_raises(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        with self.assertRaises(TypeError):
            b.build(
                GraphSpec(
                    nodes=[NodeSpec("wm", MemorySlot.WM, "stub")],
                    default_write_policy="not a dict",
                )
            )

    def test_build_unknown_impl_raises(self):
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        with self.assertRaises(KeyError):
            b.build(
                GraphSpec(
                    nodes=[NodeSpec("wm", MemorySlot.WM, "ghost")]
                )
            )

    def test_build_full_topology_4_nodes(self):
        # WM → EM (FEEDS) ; EM → SM (CONSOLIDATES_TO) ; SG → EM (INDEXES)
        r = _registry_with_stubs()
        b = MemoryGraphBuilder(r)
        g = b.build(
            GraphSpec(
                nodes=[
                    NodeSpec("wm", MemorySlot.WM, "stub"),
                    NodeSpec("em", MemorySlot.EM, "stub"),
                    NodeSpec("sm", MemorySlot.SM, "stub"),
                    NodeSpec("sg", MemorySlot.SG, "stub"),
                ],
                edges=[
                    EdgeSpec("wm", "em", EdgeKind.FEEDS),
                    EdgeSpec(
                        "em", "sm", EdgeKind.CONSOLIDATES_TO,
                        policy_type="consolidation", policy_name="passthrough",
                    ),
                    EdgeSpec("sg", "em", EdgeKind.INDEXES),
                ],
            )
        )
        self.assertEqual(len(g), 4)
        kinds = [e.kind for e in g.edges()]
        self.assertEqual(
            sorted(k.name for k in kinds),
            ["CONSOLIDATES_TO", "FEEDS", "INDEXES"],
        )


if __name__ == "__main__":
    unittest.main()
