"""End-to-end plug-in scenario.

This file exercises the *full* framework in one realistic configuration:

1. A *custom* (user-supplied) EpisodicMemory subclass is registered.
2. A 4-node graph is built declaratively (WM → EM → SM, SG INDEXES EM).
3. An observation written to WM propagates along FEEDS to EM.
4. ``run_consolidation_pass`` extracts facts from EM into SM.
5. Three different queries (planner-spatial / stopper-temporal / answerer-semantic)
   demonstrate MemoryEQA-style multi-module fan-out read.

If this test file passes, the framework is genuinely usable for downstream
EQA methods to plug into — i.e., the abstraction is not "biased" toward the
reference implementation alone.
"""
from __future__ import annotations

import unittest

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryBuilder, QueryResult
from unimem.core.slot_abc import (
    EpisodicMemoryABC,
    SceneGraphMemoryABC,
    SemanticMemoryABC,
    WorkingMemoryABC,
)
from unimem.core.slots import MemorySlot
from unimem.factory.registry import Registry
from unimem.graph.builder import EdgeSpec, GraphSpec, MemoryGraphBuilder, NodeSpec
from unimem.graph.edge import EdgeKind
from unimem.graph.graph import MemoryGraph
from unimem.policies.consolidation_policy import ConsolidationPolicy
from unimem.reference.consolidate_extract import ExtractFactsConsolidationPolicy
from unimem.reference.episodic_memory import ListEpisodicMemory
from unimem.reference.forget_fifo import FIFOForgetPolicy


# --------------------------------------------------------------------------- #
# Custom plug-in implementations (NOT the reference ones)
# --------------------------------------------------------------------------- #
class CustomWorkingMemory(WorkingMemoryABC):
    """User-supplied WM: keeps only the latest observation."""

    def __init__(self):
        self._current = None
        self.history = []

    def write(self, entry, context):
        self.history.append(entry.entry_id)
        self._current = entry
        return True

    def read(self, query):
        if self._current is None:
            return QueryResult()
        return QueryResult(entries=[self._current])

    def clear(self):
        self._current = None
        self.history.clear()

    def stats(self):
        return {"count": 1 if self._current else 0}

    def get_current(self):
        return self._current

    def set_current(self, entry):
        self._current = entry


class CustomSceneGraph(SceneGraphMemoryABC):
    """User-supplied SG: tree of objects with parent/child links."""

    def __init__(self):
        # object_id -> {attrs, parent_id}
        self._objects = {}

    def write(self, entry, context):
        # Treat entry.metadata as the object to add, if present.
        if "object_id" in entry.metadata:
            return self.add_object(
                entry.metadata["object_id"],
                entry.metadata.get("parent_id"),
                **entry.metadata.get("attrs", {}),
            )
        return False

    def read(self, query):
        # Build entries from objects matching the semantic query
        matched = []
        for oid, rec in self._objects.items():
            if not query.semantic:
                matched.append((oid, rec))
                continue
            tags = rec.get("attrs", {}).get("tags", [])
            if any(t in query.semantic for t in tags):
                matched.append((oid, rec))
        entries = [
            MemoryEntry(
                f"sg-{oid}",
                rec.get("attrs", {}).get("label", oid),
                semantic_keys=rec.get("attrs", {}).get("tags", []),
                spatial_keys=[rec["attrs"]["pos"]] if "pos" in rec.get("attrs", {}) else [],
                metadata={"object_id": oid},
            )
            for oid, rec in matched
        ]
        return QueryResult(entries=entries)

    def clear(self):
        self._objects.clear()

    def stats(self):
        return {"count": len(self._objects)}

    def add_object(self, object_id, parent_id=None, **attrs):
        if object_id in self._objects:
            return False
        self._objects[object_id] = {"attrs": attrs, "parent_id": parent_id}
        return True

    def get_children(self, parent_id):
        return [
            oid for oid, rec in self._objects.items()
            if rec.get("parent_id") == parent_id
        ]

    def get_object_by_id(self, object_id):
        rec = self._objects.get(object_id)
        return None if rec is None else {**rec["attrs"], "parent_id": rec["parent_id"]}


class CustomSemanticMemory(SemanticMemoryABC):
    """User-supplied SM: a triple store with set semantics."""

    def __init__(self):
        self._triples = set()  # (s, p, o)

    def write(self, entry, context):
        md = entry.metadata
        if "subject" in md and "predicate" in md:
            self._triples.add((md["subject"], md["predicate"], md.get("object")))
            return True
        return False

    def read(self, query):
        triples = self.query_facts()
        entries = [
            MemoryEntry(
                f"fact-{i}",
                f"({s},{p},{o})",
                semantic_keys=[s, str(p)],
                metadata={"subject": s, "predicate": p, "object": o},
            )
            for i, (s, p, o) in enumerate(triples)
        ]
        return QueryResult(entries=entries)

    def clear(self):
        self._triples.clear()

    def stats(self):
        return {"count": len(self._triples)}

    def add_fact(self, s, p, o):
        self._triples.add((s, p, o))
        return True

    def query_facts(self, subject=None, predicate=None, obj=None):
        return [
            (s, p, o)
            for (s, p, o) in self._triples
            if (subject is None or s == subject)
            and (predicate is None or p == predicate)
            and (obj is None or o == obj)
        ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_registry() -> Registry:
    r = Registry()
    r.register_module(MemorySlot.WM, "custom", CustomWorkingMemory)
    r.register_module(MemorySlot.SG, "custom", CustomSceneGraph)
    r.register_module(MemorySlot.EM, "list", ListEpisodicMemory)
    r.register_module(MemorySlot.SM, "triple", CustomSemanticMemory)
    r.register_policy(
        "consolidation", "extract_facts", ExtractFactsConsolidationPolicy
    )
    r.register_policy("forget", "fifo_100", FIFOForgetPolicy)
    return r


def _make_graph() -> MemoryGraph:
    """WM → EM (FEEDS) ; EM → SM (CONSOLIDATES_TO with extract_facts) ;
    SG → EM (INDEXES)."""
    spec = GraphSpec(
        nodes=[
            NodeSpec("wm", MemorySlot.WM, "custom"),
            NodeSpec("sg", MemorySlot.SG, "custom"),
            NodeSpec("em", MemorySlot.EM, "list", kwargs={"timescales": (60.0, 600.0)}),
            NodeSpec("sm", MemorySlot.SM, "triple"),
        ],
        edges=[
            EdgeSpec("wm", "em", EdgeKind.FEEDS),
            EdgeSpec(
                "em", "sm", EdgeKind.CONSOLIDATES_TO,
                policy_type="consolidation", policy_name="extract_facts",
            ),
            EdgeSpec("sg", "em", EdgeKind.INDEXES),
        ],
        default_forget_policy={"name": "fifo_100", "kwargs": {"capacity": 100}},
    )
    return MemoryGraphBuilder(_make_registry()).build(spec)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestPlugInScenario(unittest.TestCase):
    def setUp(self):
        self.g = _make_graph()
        self.ctx = MemoryContext(episode_id="ep1", timestamp=1.0)

    def test_graph_built_correctly(self):
        self.assertEqual(len(self.g), 4)
        kinds = [e.kind for e in self.g.edges()]
        self.assertEqual(
            sorted(k.name for k in kinds),
            ["CONSOLIDATES_TO", "FEEDS", "INDEXES"],
        )

    def test_observation_propagates_wm_to_em(self):
        obs = MemoryEntry(
            "obs1",
            "saw a red chair near the window",
            semantic_keys=["chair", "red", "window"],
            spatial_keys=[(1.5, 2.5)],
            temporal_keys=[1.0],
            source_slot="wm",
        )
        result = self.g.write(obs, self.ctx, source_node_id="wm")
        self.assertEqual(result["wm"], True)
        self.assertEqual(result["em"], True)
        # SM is not a FEEDS target, not reached
        self.assertNotIn("sm", result)
        # WM holds it as the current observation
        wm_mod = self.g.get_node("wm").module
        self.assertEqual(wm_mod.get_current().entry_id, "obs1")
        # EM stored it too
        em_mod = self.g.get_node("em").module
        em_entries = em_mod.read(Query()).entries
        self.assertEqual({e.entry_id for e in em_entries}, {"obs1"})

    def test_consolidation_extracts_facts_into_sm(self):
        # Stage some events first
        self.g.write(
            MemoryEntry(
                "obs1", "red chair",
                semantic_keys=["chair"], temporal_keys=[1.0], source_slot="wm",
            ),
            self.ctx,
            source_node_id="wm",
        )
        self.g.write(
            MemoryEntry(
                "obs2", "blue sofa",
                semantic_keys=["sofa"], temporal_keys=[2.0], source_slot="wm",
            ),
            self.ctx,
            source_node_id="wm",
        )
        # Before consolidation, SM is empty
        sm_mod = self.g.get_node("sm").module
        self.assertEqual(sm_mod.stats()["count"], 0)
        # Run consolidation
        result = self.g.run_consolidation_pass(self.ctx)
        self.assertEqual(result["stored"]["sm"], 2)
        # SM now has the two facts
        facts = sm_mod.query_facts()
        subjects = sorted(s for (s, _, _) in facts)
        self.assertEqual(subjects, ["chair", "sofa"])

    def test_planner_spatial_query(self):
        # Populate SG with two objects, then query spatially.
        sg = self.g.get_node("sg").module
        sg.add_object("room1", parent_id=None, label="living room", tags=["room"],
                      pos=(0.0, 0.0))
        sg.add_object("obj1", parent_id="room1", label="chair", tags=["chair", "red"],
                      pos=(1.0, 2.0))

        # "planner" asks for everything tagged 'chair' (spatial-semantic filter)
        q = QueryBuilder().with_slot(MemorySlot.SG).with_semantic("chair").build()
        results = self.g.read(q)
        self.assertEqual(len(results), 1)
        sg_result = results[0]
        self.assertEqual(sg_result.source_slot, "scene_graph")
        self.assertEqual(len(sg_result.entries), 1)
        self.assertEqual(sg_result.entries[0].metadata["object_id"], "obj1")

    def test_stopper_temporal_query(self):
        # "stopper" asks EM for events in a recent time window.
        em = self.g.get_node("em").module
        em.append_event(MemoryEntry(
            "e1", "old event", temporal_keys=[0.5],
        ))
        em.append_event(MemoryEntry(
            "e2", "recent event", temporal_keys=[5.0],
        ))
        q = (
            QueryBuilder()
            .with_slot(MemorySlot.EM)
            .with_temporal(4.0, 6.0)
            .build()
        )
        results = self.g.read(q)
        self.assertEqual(len(results), 1)
        em_result = results[0]
        self.assertEqual({e.entry_id for e in em_result.entries}, {"e2"})

    def test_answerer_semantic_query_across_modules(self):
        # Stage: WM has 'red' current, EM has 'red' event, SG has 'red' object.
        wm = self.g.get_node("wm").module
        wm.set_current(MemoryEntry(
            "cur", "currently looking at red chair",
            semantic_keys=["red", "chair"], temporal_keys=[10.0],
        ))
        em = self.g.get_node("em").module
        em.append_event(MemoryEntry(
            "past", "yesterday saw red chair",
            semantic_keys=["red"], temporal_keys=[1.0],
        ))
        sg = self.g.get_node("sg").module
        sg.add_object("o", parent_id=None, label="chair", tags=["red", "chair"])

        # "answerer" asks: tell me everything tagged 'red' across all memories.
        q = QueryBuilder().with_semantic("red").build()
        results = self.g.read(q)
        # We should get hits from WM, EM, and SG (SM has no 'red' facts yet)
        sources = {r.source_slot for r in results if r.entries}
        self.assertIn("working_memory", sources)
        self.assertIn("episodic", sources)
        self.assertIn("scene_graph", sources)
        # And the merged result should contain entries from each
        merged = self.g.merge_results(results)
        self.assertGreaterEqual(len(merged.entries), 3)
        self.assertEqual(merged.metadata["n_sources"], 3)

    def test_update_all_runs_per_step(self):
        # EM's default update() is a no-op; this just verifies the broadcast works.
        self.g.update_all(self.ctx)  # should not raise

    def test_full_pipeline(self):
        """Write observations, run consolidation, then query SM."""
        self.g.write(
            MemoryEntry(
                "obs1", "red chair",
                semantic_keys=["chair", "red"], temporal_keys=[1.0],
                source_slot="wm",
            ),
            self.ctx,
            source_node_id="wm",
        )
        self.g.run_consolidation_pass(self.ctx)
        sm = self.g.get_node("sm").module
        # Query SM for chair facts
        q = QueryBuilder().with_slot(MemorySlot.SM).build()
        results = self.g.read(q)
        self.assertEqual(len(results), 1)
        # The fact is in SM
        facts = sm.query_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0][0], "chair")  # subject

    def test_summary_works_with_mixed_modules(self):
        s = self.g.summary()
        self.assertEqual(s["n_nodes"], 4)
        for nid in ("wm", "sg", "em", "sm"):
            self.assertIn(nid, s["nodes"])


if __name__ == "__main__":
    unittest.main()
