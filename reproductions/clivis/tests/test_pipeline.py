"""Tests for the CLiViS pipeline skeleton."""
from __future__ import annotations

import json
import unittest

from reproductions._common.mocks import MockLLM, MockVLM
from reproductions.clivis.pipeline import CLiViSPipeline, PeriodInput


def _llm_init_response():
    """One canned LLM response covering all init needs."""
    return json.dumps({
        "persons": [{"name": "alice", "info": "host"}],
        "areas": [
            {"name": "kitchen", "info": "", "time_range": "00:00:00-00:00:30"},
            {"name": "living", "info": "", "time_range": "00:00:30-00:01:00"},
        ],
        "objects": [
            {"name": "cup", "period": "00:00:00-00:00:30"},
            {"name": "sofa", "period": "00:00:30-00:01:00"},
        ],
        "relations": [
            {"source": "kitchen", "target": "cup", "type": "CONTAINS",
             "time_range": "00:00:00-00:00:30"},
        ],
        "actions": [
            {"name": "pours_coffee", "info": "alice pours coffee",
             "time_range": "00:00:00-00:00:30", "agent": "alice", "patient": "cup"},
        ],
    })


class TestPipelineInit(unittest.TestCase):
    def test_initialises_cognitive_map(self):
        # LLM script: init returns full graph; subsequent calls return no-op update.
        llm = MockLLM(responses=[_llm_init_response()])
        vlm = MockVLM(default_response="I see alice in the kitchen")
        pipe = CLiViSPipeline(llm=llm, vlm=vlm, max_rounds=1)
        periods = [
            PeriodInput("00:00:00-00:00:30", "alice enters kitchen", "/tmp/a.mp4"),
            PeriodInput("00:00:30-00:01:00", "alice sits in living", "/tmp/b.mp4"),
        ]
        result = pipe.run("Where is alice?", periods)
        # Should have called LLM at least once for init
        self.assertGreaterEqual(llm.call_count, 1)
        # And VLM for initial perception
        self.assertGreaterEqual(vlm.call_count, 1)


class TestPipelineFinalAnswer(unittest.TestCase):
    def test_short_circuit_on_final_token(self):
        # LLM: init response, then immediate [final] answer
        llm = MockLLM(responses=[
            _llm_init_response(),
            "[final] alice is in the kitchen",
        ])
        vlm = MockVLM(default_response="ok")
        pipe = CLiViSPipeline(llm=llm, vlm=vlm, max_rounds=5)
        result = pipe.run("Where is alice?", [
            PeriodInput("00:00:00-00:00:30", "alice in kitchen")
        ])
        self.assertEqual(result.n_rounds, 0)
        self.assertEqual(result.answer, "alice is in the kitchen")
        # Should not have called VLM after final answer
        # (VLM was called once for initial perception only)
        self.assertEqual(vlm.call_count, 1)

    def test_loop_executes_instruction_then_final(self):
        # LLM script: init, then an instruction, then a JSON rationale extract,
        # then a no-op update, then a final answer
        llm = MockLLM(responses=[
            _llm_init_response(),                                          # init
            "[instruction]00:00:00-00:00:30|describe the cup",             # round 0
            '{"evidence": "alice holds the cup", "related_area": "kitchen", "related_obj": "cup"}',  # extract rationale
            '{"objects": [], "relations": [], "actions": []}',             # update cognitive map
            "[final] the cup is in the kitchen",                           # round 1 final
        ])
        vlm = MockVLM(default_response="I see the cup")
        pipe = CLiViSPipeline(llm=llm, vlm=vlm, max_rounds=5)
        result = pipe.run("Where is the cup?", [
            PeriodInput("00:00:00-00:00:30", "kitchen scene")
        ])
        self.assertEqual(result.answer, "the cup is in the kitchen")
        self.assertGreaterEqual(result.n_rationales, 1)
        # VLM was called for initial perception + once for the instruction
        self.assertEqual(vlm.call_count, 2)

    def test_max_rounds_fallback(self):
        # LLM always returns an instruction (never a final)
        llm = MockLLM(
            default_response="[instruction]00:00:00-00:00:30|look around",
            # First call is init via responses queue
            responses=[_llm_init_response()],
        )
        vlm = MockVLM(default_response="nothing new")
        pipe = CLiViSPipeline(llm=llm, vlm=vlm, max_rounds=2)
        result = pipe.run("Where?", [PeriodInput("00:00:00-00:00:30", "x")])
        self.assertEqual(result.n_rounds, 2)
        # Fallback prompt yielded an "instruction" looking string — that's fine,
        # the pipeline just returns whatever the LLM produced.
        self.assertTrue(result.answer)


class TestInstructionParsing(unittest.TestCase):
    def test_parse_pipe_format(self):
        pipe = CLiViSPipeline(llm=lambda x: "", vlm=lambda *a, **kw: "")
        period, instr = pipe._parse_instruction(
            "[instruction]00:00:00-00:00:30|describe the table"
        )
        self.assertEqual(period, "00:00:00-00:00:30")
        self.assertEqual(instr, "describe the table")

    def test_parse_without_pipe(self):
        pipe = CLiViSPipeline(llm=lambda x: "", vlm=lambda *a, **kw: "")
        period, instr = pipe._parse_instruction("[instruction]just text")
        self.assertEqual(period, "")
        self.assertEqual(instr, "just text")

    def test_no_token_returns_empty(self):
        pipe = CLiViSPipeline(llm=lambda x: "", vlm=lambda *a, **kw: "")
        period, instr = pipe._parse_instruction("no token here")
        self.assertEqual(period, "")
        self.assertEqual(instr, "")


if __name__ == "__main__":
    unittest.main()
