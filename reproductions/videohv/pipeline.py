"""VideoHV-Agent pipeline skeleton — hypothesis → distinctness → verification.

Reproduces ``VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py``
with the agent loop reading from / writing to the two synthesised memories:

* Reads clip summaries from :class:`VideoSummaryMemory`.
* Reads prior verification trace from :class:`VerificationTraceMemory`.
* Writes each round's trace into :class:`VerificationTraceMemory`.

LLM and vision tools are injectable. Tests use ``MockLLM`` /
``MockVisionTools``; production wiring passes real OpenAI / GPT-4o clients
matching the original code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .memory.time_verification_trace import VerificationTraceMemory
from .memory.video_summary_memory import VideoSummaryMemory


LLMFn = Callable[[str], str]
VisionTools = Any  # has .caption/.detect/.track


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    option: int
    text: str
    distinction_score: Optional[float] = None


@dataclass
class VerificationTrace:
    round_index: int
    clue: Optional[str] = None
    verdict: Optional[str] = None
    hypotheses: List[Hypothesis] = field(default_factory=list)
    answer_choice: Optional[int] = None


@dataclass
class VideoHVBundle:
    """The pre-computed video context bundle (matches runner.py:46-48)."""

    action_caption_summaries: List[str] = field(default_factory=list)
    object_detections_summaries: List[List[Dict[str, Any]]] = field(default_factory=list)
    clip_boundaries: List = field(default_factory=list)
    options: List[str] = field(default_factory=list)  # multiple-choice answers


@dataclass
class VideoHVResult:
    answer: int  # 0-based option index
    n_rounds: int
    trace: List[VerificationTrace]
    final_clue: Optional[str]
    verified: bool


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class VideoHVPipeline:
    """Hypothesis-verification loop with memory.

    Parameters
    ----------
    summary_memory:
        :class:`VideoSummaryMemory` instance pre-loaded with clip summaries.
        Pass ``None`` to construct from ``bundle`` on :meth:`run`.
    trace_memory:
        :class:`VerificationTraceMemory` instance. ``None`` → new empty one.
    llm:
        Callable for hypothesis generation / distinctness judging / answer
        selection.
    vision_tools:
        Object with ``.caption(frames, question)``, ``.detect(frames, query)``,
        ``.track(frames, query)`` methods. Pass ``MockVisionTools`` in tests.
    max_rounds:
        Cap on refinement rounds (default 3, matching MAX_REFINEMENT_ROUNDS).
    distinction_threshold:
        Below this score, hypotheses regenerate (default 0.5).
    """

    NOT_VERIFIED_TOKEN = "not_verified"

    def __init__(
        self,
        llm: LLMFn,
        vision_tools: Optional[VisionTools] = None,
        summary_memory: Optional[VideoSummaryMemory] = None,
        trace_memory: Optional[VerificationTraceMemory] = None,
        max_rounds: int = 3,
        distinction_threshold: float = 0.5,
        sample_frames_per_call: int = 4,
    ) -> None:
        self.llm = llm
        self.vision_tools = vision_tools
        self.summary_memory = summary_memory or VideoSummaryMemory()
        self.trace_memory = trace_memory or VerificationTraceMemory()
        self.max_rounds = int(max_rounds)
        self.distinction_threshold = float(distinction_threshold)
        self.sample_frames_per_call = int(sample_frames_per_call)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        bundle: Optional[VideoHVBundle] = None,
    ) -> VideoHVResult:
        # Lazy-ingest the summary memory if the caller supplied a bundle.
        if bundle is not None and not self.summary_memory._entries:
            self.summary_memory.ingest(
                bundle.action_caption_summaries,
                bundle.object_detections_summaries,
                bundle.clip_boundaries,
            )
        options = bundle.options if bundle else []
        if not options:
            options = ["option A", "option B"]

        trace: List[VerificationTrace] = []
        prior_lines: List[str] = []
        verification_text = ""

        for round_idx in range(self.max_rounds):
            # ---- Stage 1: hypothesis generation ----
            if round_idx == 0:
                hypotheses = self._generate_initial_hypotheses(question, options)
            else:
                hypotheses = self._regenerate_after_failure(
                    question, options, prior_lines, verification_text
                )
            prior_lines = [h.text for h in hypotheses]

            # Single-option short-circuit
            if len(hypotheses) == 1:
                answer = hypotheses[0].option
                self.trace_memory.record_round(
                    round_index=round_idx,
                    hypotheses=[h.text for h in hypotheses],
                    answer_choice=answer,
                )
                trace.append(VerificationTrace(
                    round_index=round_idx,
                    hypotheses=hypotheses,
                    answer_choice=answer,
                ))
                return VideoHVResult(
                    answer=answer, n_rounds=round_idx + 1,
                    trace=trace, final_clue=None, verified=True,
                )

            # ---- Stage 2: distinctness judging ----
            clue, distinction_score = self._judge_distinctness(hypotheses)
            if distinction_score < self.distinction_threshold:
                # Regenerate with low-distinction feedback
                hypotheses = self._regenerate_after_low_distinction(
                    question, options, [h.text for h in hypotheses], clue
                )
                prior_lines = [h.text for h in hypotheses]

            # ---- Stage 3: verification via vision tools ----
            verification_text = self._verify_with_tools(question, clue, options)
            verified = self.NOT_VERIFIED_TOKEN not in verification_text

            # Persist trace for this round
            self.trace_memory.record_round(
                round_index=round_idx,
                hypotheses=[h.text for h in hypotheses],
                clue=clue,
                distinction_score=distinction_score,
                verdict=verification_text,
            )
            trace.append(VerificationTrace(
                round_index=round_idx,
                clue=clue,
                verdict=verification_text,
                hypotheses=hypotheses,
            ))

            if verified:
                # ---- Stage 4: answer selection ----
                answer = self._select_answer(
                    question, options, hypotheses, verification_text
                )
                self.trace_memory.record_round(round_index=round_idx, answer_choice=answer)
                trace[-1].answer_choice = answer
                return VideoHVResult(
                    answer=answer, n_rounds=round_idx + 1,
                    trace=trace, final_clue=clue, verified=True,
                )

        # Exhausted rounds — pick best hypothesis by best distinction score
        # of the final round (or first option as a safe default).
        best = hypotheses[0].option if hypotheses else 0
        self.trace_memory.record_round(round_index=self.max_rounds - 1, answer_choice=best)
        if trace:
            trace[-1].answer_choice = best
        return VideoHVResult(
            answer=best, n_rounds=self.max_rounds,
            trace=trace, final_clue=trace[-1].clue if trace else None,
            verified=False,
        )

    # ------------------------------------------------------------------ #
    # Stages (each is one LLM call with a fixed prompt template)
    # ------------------------------------------------------------------ #
    def _generate_initial_hypotheses(
        self, question: str, options: List[str]
    ) -> List[Hypothesis]:
        prompt = self._format_prompt(
            "Generate a testable hypothesis for each answer option.\n"
            "Return one per line, prefixed by '<option_index>:'.\n"
            f"Question: {question}\n"
            f"Options: {options}",
        )
        return self._parse_hypotheses(self.llm(prompt), len(options))

    def _regenerate_after_failure(
        self,
        question: str,
        options: List[str],
        prior_lines: List[str],
        verification_text: str,
    ) -> List[Hypothesis]:
        prompt = self._format_prompt(
            "Regenerate more discriminative hypotheses based on the prior failed attempt.\n"
            f"Question: {question}\nOptions: {options}\n"
            f"Prior hypotheses: {prior_lines}\n"
            f"Verification feedback: {verification_text}\n"
            "Return one per line, prefixed by '<option_index>:'."
        )
        return self._parse_hypotheses(self.llm(prompt), len(options))

    def _regenerate_after_low_distinction(
        self,
        question: str,
        options: List[str],
        prior_lines: List[str],
        clue: str,
    ) -> List[Hypothesis]:
        prompt = self._format_prompt(
            "The hypotheses were insufficiently distinct. Regenerate.\n"
            f"Question: {question}\nOptions: {options}\n"
            f"Prior: {prior_lines}\nDistinguishing clue: {clue}\n"
            "Return one per line, prefixed by '<option_index>:'."
        )
        return self._parse_hypotheses(self.llm(prompt), len(options))

    def _judge_distinctness(self, hypotheses: List[Hypothesis]) -> tuple:
        prompt = self._format_prompt(
            "Judge how distinct the following hypotheses are.\n"
            "Return '<score>|<clue>' where score is in [0, 1] and clue is a "
            "short visual cue that would distinguish them.\n"
            f"Hypotheses: {[h.text for h in hypotheses]}",
        )
        out = self.llm(prompt)
        # Parse "<score>|<clue>" (default to 0.5/blank clue)
        try:
            score_str, clue = out.split("|", 1)
            score = float(score_str.strip())
        except (ValueError, AttributeError):
            score, clue = 0.5, out
        return clue.strip(), max(0.0, min(1.0, score))

    def _verify_with_tools(
        self,
        question: str,
        clue: str,
        options: List[str],
    ) -> str:
        """Run vision tools to check the clue against video evidence.

        If no vision tools are wired, the verification trivially succeeds
        (or returns not_verified if the LLM is scripted to do so).
        """
        if self.vision_tools is None:
            # Without tools we let the LLM do textual verification by
            # forwarding the clue.
            return self.llm(
                f"Verify the clue against available clip summaries.\n"
                f"Question: {question}\nClue: {clue}\n"
                f"Options: {options}\n"
                f"Summaries available: {self.summary_memory.stats()['count']}\n"
                "Reply with the verification result, or 'not_verified'."
            )
        # Sample frames from the summary memory
        summaries = self.summary_memory.get_timeline()
        frames = []
        for s in summaries[: self.sample_frames_per_call]:
            # Frames are abstract here — caller's vision_tools knows how to
            # resolve "clip-N" to real images via its own mapping.
            frames.append(s.entry_id)
        caption = self.vision_tools.caption(frames, question=clue)
        verdict = self.llm(
            f"Based on the visual evidence below, is the clue verified?\n"
            f"Clue: {clue}\nCaption: {caption}\n"
            "Reply with 'verified' or 'not_verified'."
        )
        return verdict

    def _select_answer(
        self,
        question: str,
        options: List[str],
        hypotheses: List[Hypothesis],
        verification_text: str,
    ) -> int:
        prompt = self._format_prompt(
            "Select the best-supported answer option.\n"
            f"Question: {question}\nOptions: {options}\n"
            f"Hypotheses: {[h.text for h in hypotheses]}\n"
            f"Verification: {verification_text}\n"
            "Reply with just the option index (integer).",
        )
        out = self.llm(prompt).strip()
        try:
            return int(out)
        except ValueError:
            # Take first digit found
            for ch in out:
                if ch.isdigit():
                    return int(ch)
            return 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _format_prompt(self, body: str) -> str:
        return body

    def _parse_hypotheses(self, llm_output: str, n_options: int) -> List[Hypothesis]:
        """Parse lines like ``"0: <text>"`` into Hypothesis objects.

        Falls back to one hypothesis per non-empty line.
        """
        out: List[Hypothesis] = []
        for line in llm_output.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line[:4]:
                idx_str, text = line.split(":", 1)
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    continue
                out.append(Hypothesis(option=idx, text=text.strip()))
            else:
                # Freeform line — assign next sequential index
                out.append(Hypothesis(option=len(out), text=line))
        if not out:
            # Fallback: one per option
            for i in range(n_options):
                out.append(Hypothesis(option=i, text=f"(default) option {i}"))
        return out


__all__ = [
    "VideoHVPipeline",
    "VideoHVBundle",
    "VideoHVResult",
    "Hypothesis",
    "VerificationTrace",
]
