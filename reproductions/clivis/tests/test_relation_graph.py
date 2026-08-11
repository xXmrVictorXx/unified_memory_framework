"""Tests for CLiViS RelationGraph (storage-backed property graph)."""
from __future__ import annotations

import unittest

from reproductions.clivis.memory.relation_graph import NodeLabels, RelationGraph
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage


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


class TestStorageBackedRelationGraph(unittest.TestCase):
    """Verify the rewrite delegates to GraphStorage + slot label conventions."""

    def test_default_storage_is_in_memory(self):
        rg = RelationGraph()
        self.assertIsInstance(rg.graph_storage, InMemoryGraphStorage)

    def test_nodes_get_scene_graph_slot_label(self):
        rg = RelationGraph()
        rg.add_person("alice", "host")
        node = rg.graph_storage.get_node("alice")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.SG.value, node["labels"])
        self.assertIn("Person", node["labels"])

    def test_external_storage_can_be_injected(self):
        gs = InMemoryGraphStorage()
        rg = RelationGraph(graph_storage=gs)
        rg.add_person("alice", "info")
        # Same backend
        self.assertIs(rg.graph_storage, gs)
        self.assertIsNotNone(gs.get_node("alice"))

    def test_relations_are_typed_edges(self):
        rg = RelationGraph()
        rg.add_area("kitchen", "00:00:00-00:00:30", "")
        rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        rg.add_relation("kitchen", "cup", "CONTAINS", "info", "00:00:00-00:00:30")
        # The CONTAINS edge should be queryable via Cypher
        rows = rg.graph_storage.query(
            "MATCH (a)-[r:CONTAINS]->(b) RETURN a, b"
        )
        self.assertEqual(len(rows), 1)

    def test_action_creates_performs_edge(self):
        rg = RelationGraph()
        rg.add_person("alice", "")
        rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        action_id = rg.add_action(
            action_name="pours_coffee",
            action_info="alice pours coffee",
            time_range="00:00:00-00:00:30",
            node_agent_name="alice",
            node_patient_name="cup",
        )
        # Action should have a PERFORMS edge to alice
        neighbours = rg.graph_storage.get_neighbors(action_id, "PERFORMS", "out")
        self.assertEqual(len(neighbours), 1)
        self.assertEqual(neighbours[0][0], "alice")
        # And AFFECTS edge to cup
        neighbours = rg.graph_storage.get_neighbors(action_id, "AFFECTS", "out")
        self.assertEqual(len(neighbours), 1)
        self.assertEqual(neighbours[0][0], "cup")

    def test_action_chain_via_next_action_edges(self):
        rg = RelationGraph()
        rg.add_person("alice", "")
        a1 = rg.add_action(
            "enter", "", "00:00:00-00:00:10", node_agent_name="alice"
        )
        a2 = rg.add_action(
            "sit", "", "00:00:10-00:00:20",
            node_agent_name="alice", prev_action_id=a1,
        )
        chain = rg.get_action_chain()
        self.assertEqual(chain, [a1, a2])

    def test_all_node_names(self):
        rg = RelationGraph()
        rg.add_person("alice", "")
        rg.add_area("kitchen", "00:00:00-00:00:30", "")
        rg.add_update_objects("cup", "00:00:00-00:00:30", "")
        names = set(rg.all_node_names())
        self.assertEqual(names, {"alice", "kitchen", "cup"})

    def test_stats_report_per_label_counts(self):
        rg = RelationGraph()
        rg.add_person("alice", "")
        rg.add_person("bob", "")
        rg.add_area("kitchen", "00:00:00-00:00:30", "")
        s = rg.stats()
        self.assertEqual(s["per_label"].get("Person"), 2)
        self.assertEqual(s["per_label"].get("Area"), 1)

    def test_clear_removes_all_nodes(self):
        rg = RelationGraph()
        rg.add_person("alice", "")
        rg.add_area("kitchen", "00:00:00-00:00:30", "")
        rg.clear()
        self.assertEqual(len(rg.all_node_names()), 0)


class TestCypherQueries(unittest.TestCase):
    """Verify Cypher-style queries against the storage backend work."""

    def setUp(self):
        self.rg = RelationGraph()
        self.rg.add_person("alice", "host")
        self.rg.add_area("kitchen", "00:00:00-00:00:30", "cooking area")
        self.rg.add_update_objects("cup", "00:00:00-00:00:30", "red")
        self.rg.add_relation("kitchen", "cup", "CONTAINS", "", "00:00:00-00:00:30")

    def test_match_by_label(self):
        rows = self.rg.graph_storage.query(
            f"MATCH (n:Person) RETURN n"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"]["node_id"], "alice")

    def test_match_with_where(self):
        rows = self.rg.graph_storage.query(
            f"MATCH (n:Object) WHERE n.info = 'red' RETURN n"
        )
        self.assertEqual(len(rows), 1)

    def test_match_by_edge_type(self):
        rows = self.rg.graph_storage.query(
            "MATCH (a)-[r:CONTAINS]->(b) RETURN a, b"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["a"]["node_id"], "kitchen")
        self.assertEqual(rows[0]["b"]["node_id"], "cup")


if __name__ == "__main__":
    unittest.main()
