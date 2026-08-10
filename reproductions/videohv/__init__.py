"""VideoHV-Agent reproduction — synthesising memory from a stateless pipeline.

VideoHV-Agent (CVPR 2026) is a hypothesis-verification multi-agent framework
for long-video QA. As the explore analysis confirmed, the original code has
*no* dedicated memory module — pre-computed video summaries are loaded from
static JSON, and inter-round state (verification trace, prior hypotheses)
lives in local variables.

This package reframes that picture: **the pre-computed video summaries ARE
episodic memory**, indexed by clip / time / object. The hypothesis-
verification loop is the *pipeline* that reads from this memory. We expose:

* :class:`VideoSummaryMemory` — episodic store of clip-level summaries
  (action captions + object detections + clip boundaries), implementing
  :class:`~unimem.core.slot_abc.EpisodicMemoryABC`.
* :class:`VerificationTraceMemory` — short-term episodic log of prior
  hypotheses / verdicts / clues produced across refinement rounds.
* :class:`VideoHVPipeline` — the hypothesis → distinctness → verification →
  answer loop with injectable LLM + vision tools. Reads from both memories
  on every round; writes the verification trace.

The framing question: *"Can the unified framework support a method that
wasn't designed with explicit memory?"* The answer is yes — the framework's
multi-axis index fits the clip-summary structure naturally, and reframing
the inter-round state as a second episodic memory makes the agent loop
inspectable via standard unimem reads.
"""
from __future__ import annotations

from .memory.time_verification_trace import VerificationTraceMemory
from .memory.video_summary_memory import VideoSummaryMemory
from .pipeline import (
    Hypothesis,
    VideoHVBundle,
    VideoHVPipeline,
    VideoHVResult,
    VerificationTrace,
)

__all__ = [
    "VideoSummaryMemory",
    "VerificationTraceMemory",
    "VideoHVPipeline",
    "VideoHVBundle",
    "VideoHVResult",
    "Hypothesis",
    "VerificationTrace",
]
