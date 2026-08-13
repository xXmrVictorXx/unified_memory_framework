"""Tests for CLiViS TimeWorkingMemory (storage-backed)."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockLLM
from reproductions.clivis.memory.time_working_memory import (
    Rationale,
    TimeWorkingMemory,
    _period_to_seconds,
)
from reproductions.llm import safe_json_extract
from unimem.core.entry import MemoryEntry
from unimem.core.slots import MemorySlot
from unimem.graph_storage import InMemoryGraphStorage


class TestRationale(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        r = Rationale("evidence text", "kitchen", "00:00:10-00:00:20", "cup")
        d = r.to_dict()
        self.assertEqual(d["evidence"], "evidence text")
        self.assertEqual(d["related_obj"], "cup")


class TestPeriodParsing(unittest.TestCase):
    def test_parse_period(self):
        self.assertEqual(_period_to_seconds("00:00:10-00:00:30"), 20.0)
        self.assertEqual(_period_to_seconds("00:01:00"), 60.0)
        self.assertIsNone(_period_to_seconds(""))
        self.assertIsNone(_period_to_seconds("invalid"))


class TestSafeJsonExtract(unittest.TestCase):
    def test_direct_json(self):
        self.assertEqual(safe_json_extract('{"a": 1}'), {"a": 1})

    def test_json_in_text(self):
        self.assertEqual(safe_json_extract('ok {"a": 1} done'), {"a": 1})

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(safe_json_extract(text), {"a": 1})

    def test_invalid_returns_none(self):
        self.assertIsNone(safe_json_extract("not json at all"))
        self.assertIsNone(safe_json_extract(""))


class TestTimeWorkingMemoryBasic(unittest.TestCase):
    def test_constructor_defaults(self):
        m = TimeWorkingMemory(question="where is the cup?")
        self.assertEqual(m.question, "where is the cup?")
        self.assertEqual(m.history_messages, [])
        self.assertEqual(m.rationale_list, [])
        self.assertIsInstance(m.graph_storage, InMemoryGraphStorage)

    def test_update_history_msg_appends_pair(self):
        m = TimeWorkingMemory(question="q")
        m.update_history_msg("look at the table", "I see a cup")
        self.assertEqual(len(m.history_messages), 2)
        self.assertEqual(m.history_messages[0]["role"], "assistant")
        self.assertEqual(m.history_messages[1]["role"], "user")

    def test_extract_and_update_rationale_list_parses_json(self):
        llm = MockLLM(
            default_response='{"evidence": "cup is on table", "related_area": "kitchen", "related_obj": "cup"}'
        )
        m = TimeWorkingMemory(question="where is the cup?", llm_extractor=llm)
        m.update_history_msg("look", "I see a cup")
        rat = m.extract_and_update_rationale_list("00:00:10-00:00:20")
        self.assertIsNotNone(rat)
        self.assertEqual(rat.evidence, "cup is on table")
        self.assertEqual(rat.related_obj, "cup")
        self.assertEqual(m.get_rationale_count(), 1)

    def test_extract_returns_none_on_empty_evidence(self):
        llm = MockLLM(default_response='{"evidence": ""}')
        m = TimeWorkingMemory(question="q", llm_extractor=llm)
        m.update_history_msg("look", "nothing here")
        self.assertIsNone(m.extract_and_update_rationale_list("p"))

    def test_extract_requires_llm(self):
        m = TimeWorkingMemory(question="q")
        m.update_history_msg("a", "b")
        with self.assertRaises(RuntimeError):
            m.extract_and_update_rationale_list("p")

    def test_extract_noop_without_messages(self):
        m = TimeWorkingMemory(question="q", llm_extractor=MockLLM())
        self.assertIsNone(m.extract_and_update_rationale_list("p"))


class TestTimeWorkingMemoryStorage(unittest.TestCase):
    def setUp(self):
        self.m = TimeWorkingMemory(question="where is the cup?")
        # Trigger rationale creation via the LLM extractor.
        self.m.update_history_msg("look", "I see a cup on the table")
        llm = MockLLM(
            default_response='{"evidence": "cup on table", "related_area": "kitchen", "related_obj": "cup"}'
        )
        self.m._llm_extractor = llm  # noqa: SLF001 — test wiring
        self.m.extract_and_update_rationale_list("00:00:05-00:00:10")

    def test_rationale_persisted_to_storage(self):
        node = self.m.graph_storage.get_node("rationale-0")
        self.assertIsNotNone(node)
        self.assertIn(MemorySlot.WM.value, node["labels"])
        self.assertIn("Rationale", node["labels"])

    def test_rationale_has_time_index_attached(self):
        """CLiViS period anchors rationales to TimeIndex nodes."""
        neighbours = self.m.graph_storage.get_neighbors(
            "rationale-0", "AT_TIME", "out"
        )
        self.assertEqual(len(neighbours), 1)
        ti_node = self.m.graph_storage.get_node(neighbours[0][0])
        self.assertIsNotNone(ti_node)
        self.assertIn("TimeIndex", ti_node["labels"])
        self.assertEqual(
            ti_node["properties"]["period"], "00:00:05-00:00:10"
        )

    def test_stats_track_rationale_count(self):
        s = self.m.stats()
        self.assertEqual(s["count"], 1)
        self.assertTrue(s["has_question"])

    def test_output_memory_info_serialises_rationales(self):
        text = self.m.output_memory_info()
        self.assertIn("Question:", text)
        self.assertIn("Rationales:", text)
        self.assertIn("cup on table", text)
        self.assertIn("kitchen", text)

    def test_clear_resets_state(self):
        self.m.clear()
        self.assertEqual(self.m.stats()["count"], 0)
        self.assertEqual(self.m.question, "")


if __name__ == "__main__":
    unittest.main()
