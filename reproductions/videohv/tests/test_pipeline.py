"""Tests for the VideoHV-Agent pipeline skeleton."""
from __future__ import annotations

import unittest

from reproductions._common.mocks import MockLLM, MockVisionTools
from reproductions.videohv.memory.time_verification_trace import VerificationTraceMemory
from reproductions.videohv.memory.video_summary_memory import VideoSummaryMemory
from reproductions.videohv.pipeline import VideoHVBundle, VideoHVPipeline


class TestPipelineSingleOption(unittest.TestCase):
    def test_short_circuit_when_one_hypothesis(self):
        llm = MockLLM(default_response="0: only one hypothesis")
        pipe = VideoHVPipeline(llm=llm)
        result = pipe.run("What?", VideoHVBundle(options=["A", "B"]))
        self.assertTrue(result.verified)
        self.assertEqual(result.answer, 0)
        self.assertEqual(result.n_rounds, 1)


class TestPipelineVerifiedRound1(unittest.TestCase):
    def test_verified_immediately(self):
        # Hypotheses (2 lines), then distinctness (0.9|some clue), then
        # verification returns "verified", then answer selection "1".
        llm = MockLLM(responses=[
            "0: alice cooked\n1: bob cooked",  # initial hypotheses
            "0.9|who is at the stove",          # distinctness
            "verified",                          # verification (no vision tools → LLM call)
            "1",                                 # answer selection
        ])
        pipe = VideoHVPipeline(llm=llm)
        result = pipe.run("Who cooked?", VideoHVBundle(options=["alice", "bob"]))
        self.assertTrue(result.verified)
        self.assertEqual(result.answer, 1)
        self.assertEqual(result.n_rounds, 1)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].clue, "who is at the stove")


class TestPipelineLowDistinctnessRegenerate(unittest.TestCase):
    def test_low_distinction_triggers_regenerate(self):
        # Initial hypotheses (2 lines)
        # Distinctness: 0.2|bad clue (below threshold)
        # Regeneration after low distinction: 2 new lines
        # Verification: "verified"
        # Answer: "0"
        llm = MockLLM(responses=[
            "0: vague hypothesis a\n1: vague hypothesis b",
            "0.2|bad clue",
            "0: refined hyp a\n1: refined hyp b",
            "verified",
            "0",
        ])
        pipe = VideoHVPipeline(llm=llm, distinction_threshold=0.5)
        result = pipe.run("Q?", VideoHVBundle(options=["a", "b"]))
        self.assertTrue(result.verified)
        self.assertEqual(result.answer, 0)
        # Trace should record regenerated hypotheses
        self.assertEqual(len(result.trace), 1)
        # Last trace should have the *refined* hypotheses text
        hyp_texts = [h.text for h in result.trace[0].hypotheses]
        self.assertIn("refined hyp a", hyp_texts)


class TestPipelineNotVerifiedRetries(unittest.TestCase):
    def test_not_verified_then_verified_next_round(self):
        # Round 0:
        #   initial hyp (2 lines)
        #   distinctness (0.8|clue0)
        #   verification: "not_verified"
        # Round 1:
        #   regenerate: 2 lines
        #   distinctness: 0.9|clue1
        #   verification: "verified"
        #   answer: "1"
        llm = MockLLM(responses=[
            "0: h0a\n1: h0b",
            "0.8|clue0",
            "not_verified",
            "0: h1a\n1: h1b",
            "0.9|clue1",
            "verified",
            "1",
        ])
        pipe = VideoHVPipeline(llm=llm)
        result = pipe.run("Q?", VideoHVBundle(options=["a", "b"]))
        self.assertTrue(result.verified)
        self.assertEqual(result.answer, 1)
        self.assertEqual(result.n_rounds, 2)
        self.assertEqual(len(result.trace), 2)

    def test_exhausted_rounds_returns_unverified(self):
        # Always returns not_verified in the verification stage.
        # Hypotheses start as 2 lines, distinctness OK, verification fails.
        llm = MockLLM(responses=[])
        # Use pattern_map to handle all stage prompts
        llm = MockLLM(
            pattern_map=[
                (r"Generate a testable", "0: hyp0\n1: hyp1"),
                (r"Judge how distinct", "0.9|clue"),
                (r"Verify the clue|is the clue verified", "not_verified"),
                (r"Regenerate", "0: new0\n1: new1"),
            ],
            default_response="not_verified",
        )
        pipe = VideoHVPipeline(llm=llm, max_rounds=2)
        result = pipe.run("Q?", VideoHVBundle(options=["a", "b"]))
        self.assertFalse(result.verified)
        self.assertEqual(result.n_rounds, 2)


class TestPipelineWithVisionTools(unittest.TestCase):
    def test_vision_tools_called(self):
        tools = MockVisionTools()
        # Hypotheses OK, distinctness OK, caption returns text, verdict "verified", answer
        llm = MockLLM(responses=[
            "0: h0\n1: h1",
            "0.8|the clue",
            "verified",  # this is the post-caption verdict from LLM
            "0",
        ])
        pipe = VideoHVPipeline(llm=llm, vision_tools=tools)
        # Pre-load some summaries so the verify path has frames to "sample"
        pipe.summary_memory.ingest(["clip A", "clip B", "clip C"])
        result = pipe.run("Q?", VideoHVBundle(options=["a", "b"]))
        self.assertTrue(result.verified)
        # The caption tool should have been called once
        self.assertEqual(len(tools.calls["caption"]), 1)


class TestPipelinePersistsTrace(unittest.TestCase):
    def test_trace_memory_records_each_round(self):
        llm = MockLLM(responses=[
            "0: hyp0\n1: hyp1",
            "0.9|clue",
            "not_verified",
            "0: hyp0v2\n1: hyp1v2",
            "0.9|clue2",
            "verified",
            "1",
        ])
        trace_mem = VerificationTraceMemory()
        pipe = VideoHVPipeline(llm=llm, trace_memory=trace_mem)
        result = pipe.run("Q?", VideoHVBundle(options=["a", "b"]))
        self.assertEqual(result.n_rounds, 2)
        # Trace memory should have both rounds
        self.assertEqual(trace_mem.n_rounds, 2)
        r0 = trace_mem.get_round(0)
        self.assertEqual(r0["verdict"], "not_verified")
        self.assertEqual(r0["clue"], "clue")
        r1 = trace_mem.get_round(1)
        self.assertEqual(r1["verdict"], "verified")
        self.assertEqual(r1["answer_choice"], 1)


class TestHypothesisParser(unittest.TestCase):
    def test_parse_prefixed_lines(self):
        pipe = VideoHVPipeline(llm=lambda x: "")
        out = pipe._parse_hypotheses("0: first\n1: second\n", 2)
        self.assertEqual([h.option for h in out], [0, 1])
        self.assertEqual(out[0].text, "first")

    def test_parse_freeform_fallback(self):
        pipe = VideoHVPipeline(llm=lambda x: "")
        out = pipe._parse_hypotheses("just text\nanother line", 2)
        self.assertEqual(len(out), 2)

    def test_parse_empty_returns_defaults(self):
        pipe = VideoHVPipeline(llm=lambda x: "")
        out = pipe._parse_hypotheses("", 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].option, 0)


if __name__ == "__main__":
    unittest.main()
