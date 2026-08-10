"""Reference implementations and policies.

This package contains:

* :class:`ListEpisodicMemory` — the single reference module, exercising
  multi-axis indexing, multi-timescale bucketing, write gating,
  consolidation, and FIFO forgetting. Designed to stress-test every part of
  the abstract contract.
* :class:`FIFOForgetPolicy` — capacity-bounded FIFO forgetting.
* :class:`ExtractFactsConsolidationPolicy` — EM→SM sedimentation that
  extracts ``(subject, was_seen, time_window)`` triples from episodic events.

These are intentionally simple — they exist to validate the ABC design, not
to be SOTA. Downstream systems are expected to swap them out.
"""
from __future__ import annotations

from .consolidate_extract import ExtractFactsConsolidationPolicy
from .episodic_memory import ListEpisodicMemory
from .forget_fifo import FIFOForgetPolicy

__all__ = [
    "ListEpisodicMemory",
    "FIFOForgetPolicy",
    "ExtractFactsConsolidationPolicy",
]
