"""Tests for CLiViS TimeWorkingMemory."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockLLM
from reproductions.clivis.memory.time_working_memory import (
    Rationale,
    TimeWorkingMemory,
    _period_to_seconds,
    _safe_json_extract,
)
from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import QueryBuilder
from unimem.core.slot_abc import EpisodicMemoryABC, WorkingMemoryABC


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
        self.assertEqual(_safe_json_extract('{"a": 1}'), {"a": 1})

    def test_json_in_text(self):
        self.assertEqual(_safe_json_extract('ok {"a": 1} done'), {"a": 1})

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(_safe_json_extract(text), {"a": 1})

    def test_invalid_returns_none(self):
        self.assertIsNone(_safe_json_extract("not json at all"))
        self.assertIsNone(_safe_json_extract(""))


class TestTimeWorkingMemoryABC(unittest.TestCase):
    def test_is_working_and_episodic(self):
        m = TimeWorkingMemory(question="q")
        self.assertIsInstance(m, WorkingMemoryABC)
        self.assertIsInstance(m, EpisodicMemoryABC)

    def test_get_current_returns_question_when_empty(self):
        m = TimeWorkingMemory(question="where is the cup?")
        cur = m.get_current()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.text, "where is the cup?")

    def test_get_current_returns_latest_rationale(self):
        m = TimeWorkingMemory(question="q")
        m.append_event(MemoryEntry(
            "r1", "evidence", metadata={"related_area": "kitchen", "related_period": "00:00:05-00:00:10"}
        ))
        cur = m.get_current()
        self.assertEqual(cur.entry_id, "rationale-0")

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


class TestTimeWorkingMemoryReadWrite(unittest.TestCase):
    def setUp(self):
        self.m = TimeWorkingMemory(question="where is the cup?")
        self.m.append_event(MemoryEntry(
            "r1", "cup on table",
            metadata={"related_area": "kitchen", "related_period": "00:00:05-00:00:10", "related_obj": "cup"}
        ))
        self.m.append_event(MemoryEntry(
            "r2", "sofa in living room",
            metadata={"related_area": "living", "related_period": "00:00:15-00:00:20", "related_obj": "sofa"}
        ))

    def test_write_appends_rationale(self):
        ok = self.m.write(MemoryEntry(
            "r3", "evidence",
            metadata={"related_area": "a", "related_period": "00:00:25-00:00:30"}
        ), MemoryContext())
        self.assertTrue(ok)
        self.assertEqual(self.m.get_rationale_count(), 3)

    def test_read_returns_all_when_no_filter(self):
        result = self.m.read(QueryBuilder().build())
        self.assertEqual(len(result.entries), 2)

    def test_read_filters_by_semantic(self):
        # AND semantics: kitchen AND cup → only r1
        result = self.m.read(QueryBuilder().with_semantic("kitchen", "cup").build())
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].entry_id, "rationale-0")

    def test_get_timeline_filters_by_time(self):
        # r1 has midpoint 7.5s; r2 has midpoint 17.5s
        timeline = self.m.get_timeline(t_min=10, t_max=20)
        self.assertEqual({e.entry_id for e in timeline}, {"rationale-1"})

    def test_output_memory_info_serialises_rationales(self):
        text = self.m.output_memory_info()
        self.assertIn("Question:", text)
        self.assertIn("Rationales:", text)
        self.assertIn("cup on table", text)
        self.assertIn("kitchen", text)

    def test_clear(self):
        self.m.clear()
        self.assertEqual(self.m.stats()["count"], 0)
        self.assertEqual(self.m.question, "")


if __name__ == "__main__":
    unittest.main()
