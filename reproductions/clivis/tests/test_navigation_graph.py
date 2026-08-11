"""Tests for CLiViS NavigationGraph (storage-backed)."""
from __future__ import annotations

import unittest

from reproductions.clivis.memory.navigation_graph import (
    NavigationGraph,
    _parse_range_seconds,
    _ranges_overlap,
)
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage


class TestRangeParsing(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(_parse_range_seconds("00:00:10-00:00:30"), (10.0, 30.0))
        self.assertIsNone(_parse_range_seconds("invalid"))

    def test_overlap(self):
        self.assertTrue(_ranges_overlap((10, 30), (20, 40)))
        self.assertFalse(_ranges_overlap((10, 20), (30, 40)))
        self.assertTrue(_ranges_overlap((10, 30), (10, 30)))  # touching


class TestNavigationGraph(unittest.TestCase):
    def setUp(self):
        self.nav = NavigationGraph(period_description_dict={
            "00:00:00-00:00:30": "opening scene",
            "00:00:30-00:01:00": "next part",
        })

    def test_init_periods(self):
        self.assertEqual(len(self.nav.get_period_names()), 2)
        self.assertIn("00:00:00-00:00:30", self.nav.periods_infos)

    def test_add_persons(self):
        self.nav.add_persons([{"name": "alice", "info": "host"}])
        self.assertIn("alice", self.nav.person_names)
        self.assertEqual(self.nav.get_person_info("alice"), "host")
        self.assertIsNone(self.nav.get_person_info("bob"))

    def test_add_areas_overlapping_periods(self):
        # area active in 00:00:20-00:00:40 overlaps both periods
        self.nav.add_areas([{"name": "kitchen", "info": "", "time_range": "00:00:20-00:00:40"}])
        self.assertIn("kitchen", self.nav.periods_to_areas["00:00:00-00:00:30"])
        self.assertIn("kitchen", self.nav.periods_to_areas["00:00:30-00:01:00"])

    def test_add_areas_no_overlap(self):
        self.nav.add_areas([{"name": "garage", "info": "", "time_range": "00:05:00-00:06:00"}])
        self.assertEqual(self.nav.area_names, {"garage"})
        for period_areas in self.nav.periods_to_areas.values():
            self.assertNotIn("garage", period_areas)

    def test_add_objs(self):
        self.nav.add_objs(["cup", "plate"], "00:00:00-00:00:30")
        self.nav.add_objs(["cup"], "00:00:00-00:00:30")  # dedup
        self.assertEqual(self.nav.periods_to_obj_names["00:00:00-00:00:30"], ["cup", "plate"])
        self.nav.add_objs(["sofa"], "00:00:30-00:01:00")
        self.assertIn("sofa", self.nav.obj_names)

    def test_add_activity(self):
        self.nav.add_activity("cooking", "00:00:00-00:00:30")
        self.assertEqual(self.nav.periods_to_activities["00:00:00-00:00:30"], "cooking")

    def test_get_entities_in_period(self):
        self.nav.add_objs(["cup"], "00:00:00-00:00:30")
        self.nav.add_areas([{"name": "kitchen", "info": "", "time_range": "00:00:10-00:00:20"}])
        ents = self.nav.get_entities_in_period("00:00:00-00:00:30")
        self.assertIn("cup", ents["objects"])
        self.assertIn("kitchen", ents["areas"])

    def test_register_video_segment(self):
        self.nav.register_video_segment("00:00:00-00:00:30", "/tmp/seg1.mp4")
        self.assertEqual(self.nav.video_segments_to_files["00:00:00-00:00:30"], "/tmp/seg1.mp4")

    def test_output_periods_info_has_content(self):
        self.nav.add_objs(["cup"], "00:00:00-00:00:30")
        text = self.nav.output_periods_info()
        self.assertIn("Period:", text)
        self.assertIn("cup", text)


class TestNavigationGraphStorage(unittest.TestCase):
    def test_external_storage_can_be_injected(self):
        gs = InMemoryGraphStorage()
        nav = NavigationGraph(graph_storage=gs)
        self.assertIs(nav.graph_storage, gs)

    def test_period_nodes_get_slot_label(self):
        nav = NavigationGraph(period_description_dict={
            "00:00:00-00:00:30": "scene"
        })
        node = nav.graph_storage.get_node("00:00:00-00:00:30")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.GM.value, node["labels"])
        self.assertIn("Period", node["labels"])

    def test_persons_persisted_to_storage(self):
        nav = NavigationGraph()
        nav.add_persons([{"name": "alice", "info": "host"}])
        node = nav.graph_storage.get_node("alice")
        self.assertIsNotNone(node)
        self.assertIn("Person", node["labels"])

    def test_object_period_edge(self):
        nav = NavigationGraph(period_description_dict={
            "00:00:00-00:00:30": "scene"
        })
        nav.add_objs(["cup"], "00:00:00-00:00:30")
        # HAS_OBJECT edge from period → object
        rows = nav.graph_storage.query(
            "MATCH (p)-[r:HAS_OBJECT]->(o) RETURN p, o"
        )
        self.assertEqual(len(rows), 1)


class TestNavigationGraphClear(unittest.TestCase):
    def test_clear(self):
        nav = NavigationGraph(period_description_dict={"p": "d"})
        nav.add_persons([{"name": "alice"}])
        nav.clear()
        self.assertEqual(len(nav.periods_infos), 0)
        self.assertEqual(len(nav.person_names), 0)


if __name__ == "__main__":
    unittest.main()
