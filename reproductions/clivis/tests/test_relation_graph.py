"""Tests for CLiViS RelationGraph (pure-Python property graph)."""
from __future__ import annotations

import unittest

from reproductions.clivis.memory.relation_graph import NodeLabels, RelationGraph
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import QueryBuilder
from unimem.core.slot_abc import SceneGraphMemoryABC, SemanticMemoryABC


class TestNodeLabels(unittest.TestCase):
    def test_from_value(self):
        self.assertEqual(NodeLabels.from_value("Person"), NodeLabels.PERSON)
        self.assertEqual(NodeLabels.from_value("person"), NodeLabels.PERSON)
        self.assertEqual(NodeLabels.from_value("PERSON"), NodeLabels.PERSON)
        with self.assertRaises(KeyError):
            NodeLabels.from_value("bogus")


class TestNodeOps(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()

    def test_add_person(self):
        self.assertTrue(self.rg.add_person("alice", "the host"))
        info = self.rg.get_node_info("alice")
        self.assertEqual(info["label"], "Person")
        self.assertEqual(info["props"]["info"], "the host")

    def test_add_object(self):
        self.assertTrue(self.rg.add_update_objects("cup", "00:00:00-00:00:30", "red"))
        info = self.rg.get_node_info("cup")
        self.assertEqual(info["label"], "Object")
        self.assertEqual(info["props"]["time_range"], "00:00:00-00:00:30")

    def test_add_area(self):
        self.assertTrue(self.rg.add_area("kitchen", "00:00:00-00:00:30", "cooking area"))
        self.assertEqual(self.rg.get_node_info("kitchen")["label"], "Area")

    def test_add_update_node_rejects_bad_label(self):
        self.assertFalse(self.rg.add_update_node("x", "NotARealLabel", {}))

    def test_update_existing_node_merges_props(self):
        self.rg.add_person("alice", "v1")
        self.rg.add_person("alice", "v2")
        info = self.rg.get_node_info("alice")
        self.assertEqual(info["props"]["info"], "v2")

    def test_get_node_info_with_label_filter(self):
        self.rg.add_person("alice", "info")
        self.assertIsNone(self.rg.get_node_info("alice", "Object"))
        self.assertIsNotNone(self.rg.get_node_info("alice", "Person"))


class TestRelationOps(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        self.rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        self.rg.add_relation("kitchen", "cup", "CONTAINS", "kitchen has cup", "00:00:00-00:00:30")

    def test_count_triples(self):
        self.assertEqual(self.rg.count_triples(), 1)

    def test_get_relations_of_node(self):
        rels = self.rg.get_relations_of_node("kitchen")
        self.assertEqual(len(rels["outgoing"]), 1)
        self.assertEqual(rels["outgoing"][0]["type"], "CONTAINS")
        rels_cup = self.rg.get_relations_of_node("cup")
        self.assertEqual(len(rels_cup["incoming"]), 1)

    def test_add_relation_rejects_unknown_endpoint(self):
        self.assertFalse(self.rg.add_relation("kitchen", "ghost", "X", "", "", None))

    def test_get_paths_between_nodes(self):
        self.assertEqual(len(self.rg.get_paths_between_nodes("kitchen", "cup")), 1)
        # Reverse direction needs dual_direction=True (default)
        self.assertEqual(len(self.rg.get_paths_between_nodes("cup", "kitchen")), 1)
        # And disabling dual_direction blocks reverse
        self.assertEqual(
            len(self.rg.get_paths_between_nodes("cup", "kitchen", dual_direction=False)), 0
        )

    def test_get_paths_respects_max_step(self):
        self.assertEqual(len(self.rg.get_paths_between_nodes("kitchen", "cup", max_step=0)), 0)


class TestActionOps(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()
        self.rg.add_person("alice", "")
        self.rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        self.rg.add_update_objects("plate", "00:00:00-00:00:30", "")
        self.aid1 = self.rg.add_action(
            "pick_up", "alice picks up cup",
            "00:00:00-00:00:30",
            node_agent_name="alice",
            node_patient_name="cup",
        )
        self.aid2 = self.rg.add_action(
            "place", "alice places cup on plate",
            "00:00:30-00:01:00",
            node_agent_name="alice",
            node_patient_name="cup",
            node_target_name="plate",
            prev_action_id=self.aid1,
        )

    def test_action_creates_activity_node(self):
        info = self.rg.get_node_info(self.aid1)
        self.assertEqual(info["label"], "Activity")

    def test_action_links_roles(self):
        rels = self.rg.get_relations_of_node(self.aid1)
        types = [r["type"] for r in rels["outgoing"]]
        self.assertIn("PERFORMS", types)  # → alice
        self.assertIn("AFFECTS", types)   # → cup

    def test_action_chain(self):
        chain = self.rg.get_action_chain(self.aid1)
        self.assertEqual(chain, [self.aid1, self.aid2])

    def test_get_action_with_relations(self):
        a = self.rg.get_action_with_relations(self.aid2)
        self.assertEqual(a["prev_action_id"], self.aid1)
        self.assertEqual(a["next_action_id"], None)
        self.assertIn("AFFECTS", [r["type"] for r in a["relations"]["outgoing"]])

    def test_get_actions_related_to_entity(self):
        actions = self.rg.get_actions_related_to_entity("alice")
        self.assertEqual(len(actions), 2)

    def test_get_actions_in_period(self):
        # action 1 time_range is [0,30]; action 2 is [30,60]. Our overlap rule
        # treats edge-touching as overlap, so a query for [0,30] matches both.
        a = self.rg.get_actions_in_period("00:00:00-00:00:30")
        self.assertEqual(len(a), 2)
        # Strict-range query (open upper bound via non-overlap):
        a = self.rg.get_actions_in_period("00:00:00-00:00:15")
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["action_id"], self.aid1)
        # Filtering by agent
        a = self.rg.get_actions_in_period("00:00:00-00:01:00", agent_name="alice")
        self.assertEqual(len(a), 2)


class TestSubgraphExtraction(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        self.rg.add_area("living", "00:00:00-00:00:30", "")
        self.rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        self.rg.add_relation("kitchen", "cup", "CONTAINS", "", "00:00:00-00:00:30")

    def test_extract_subgraph(self):
        sub = self.rg.extract_subgraph_by_nodes(["kitchen", "cup"])
        self.assertEqual(len(sub["key_nodes"]), 2)
        self.assertEqual(len(sub["paths"]), 1)

    def test_format_subgraph_text(self):
        sub = self.rg.extract_subgraph_by_nodes(["kitchen", "cup"])
        text = self.rg.format_subgraph(sub)
        self.assertIn("Key Nodes:", text)
        self.assertIn("Paths:", text)

    def test_format_subgraph_json(self):
        sub = self.rg.extract_subgraph_by_nodes(["cup"])
        j = self.rg.format_subgraph_json(sub)
        self.assertIn("key_nodes", j)


class TestSceneGraphABC(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()

    def test_is_scene_graph_abc(self):
        self.assertIsInstance(self.rg, SceneGraphMemoryABC)

    def test_add_object_with_parent(self):
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        self.assertTrue(self.rg.add_object("cup", parent_id="kitchen"))
        self.assertEqual(self.rg.get_children("kitchen"), ["cup"])

    def test_get_children_roots(self):
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        self.rg.add_object("cup", parent_id="kitchen")
        # Roots = nodes without incoming CONTAINS
        roots = self.rg.get_children(None)
        self.assertIn("kitchen", roots)
        self.assertNotIn("cup", roots)

    def test_get_object_by_id(self):
        self.rg.add_object("cup", label="Object", color="red")
        info = self.rg.get_object_by_id("cup")
        self.assertEqual(info["color"], "red")


class TestSemanticMemoryABC(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()
        self.rg.add_person("alice", "")
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")

    def test_is_semantic_abc(self):
        self.assertIsInstance(self.rg, SemanticMemoryABC)

    def test_add_fact_and_query(self):
        self.rg.add_fact("alice", "located_in", "kitchen")
        facts = self.rg.query_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0], ("alice", "located_in", "kitchen"))

    def test_query_with_filters(self):
        self.rg.add_fact("alice", "located_in", "kitchen")
        self.rg.add_fact("alice", "holds", "cup")
        self.assertEqual(len(self.rg.query_facts(predicate="located_in")), 1)
        self.assertEqual(len(self.rg.query_facts(subject="alice")), 2)
        # Filter by obj
        self.assertEqual(len(self.rg.query_facts(obj="cup")), 1)


class TestMemoryEntryBridge(unittest.TestCase):
    def setUp(self):
        self.rg = RelationGraph()

    def test_write_node(self):
        e = MemoryEntry(
            "n1", "info text",
            metadata={"kind": "node", "name": "alice", "label": "Person"},
        )
        self.assertTrue(self.rg.write(e, MemoryContext()))
        self.assertEqual(self.rg.get_node_info("alice")["label"], "Person")

    def test_write_relation(self):
        self.rg.add_update_node("a", "Area")
        self.rg.add_update_node("b", "Object")
        e = MemoryEntry(
            "r1", "rel info",
            metadata={"kind": "relation", "source": "a", "target": "b", "type": "CONTAINS"},
        )
        self.assertTrue(self.rg.write(e, MemoryContext()))
        self.assertEqual(self.rg.count_triples(), 1)

    def test_read_no_filter_returns_all_nodes(self):
        self.rg.add_person("alice", "")
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        result = self.rg.read(QueryBuilder().build())
        self.assertEqual(len(result.entries), 2)

    def test_read_with_semantic_returns_matching_facts(self):
        self.rg.add_person("alice", "")
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "")
        self.rg.add_relation("alice", "kitchen", "located_in", "", "00:00:00-00:00:30")
        result = self.rg.read(QueryBuilder().with_semantic("located_in").build())
        self.assertEqual(len(result.entries), 1)
        self.assertIn("located_in", result.entries[0].text)


if __name__ == "__main__":
    unittest.main()
