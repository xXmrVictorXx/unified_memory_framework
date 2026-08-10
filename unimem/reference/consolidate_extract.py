"""``ExtractFactsConsolidationPolicy`` — reference EM→SM sedimentation policy.

Extracts structured ``(subject, was_seen, time_window)`` facts from each
episodic event in the source memory, producing one
:class:`~unimem.core.entry.MemoryEntry` per fact ready to be ingested by a
:class:`~unimem.core.slot_abc.SemanticMemoryABC` target.

The extraction is intentionally simple (subject = first semantic key if any,
object = the entry text), to demonstrate the *shape* of a sedimentation
policy without committing to any particular NLP technique. Real systems
swap in their own LLM- or rule-based extractor.
"""
from __future__ import annotations

import itertools
from typing import List, Optional

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry
from ..core.module import MemoryModule
from ..core.query import Query
from ..policies.consolidation_policy import ConsolidationPolicy

# A monotonically increasing counter to keep extracted entry ids unique
# across calls (within a single process).
_fact_counter = itertools.count(1)


class ExtractFactsConsolidationPolicy(ConsolidationPolicy):
    """Turn episodic events into ``(subject, was_seen, time_window)`` facts.

    Parameters
    ----------
    time_window:
        Size (seconds) of the time window attached to each extracted fact.
        Defaults to the source memory's largest timescale if it exposes one,
        otherwise 600 seconds.
    only_with_semantic:
        If True (default), only events with at least one semantic key are
        extracted. Set False to extract every event (subject becomes
        ``"something"``).
    """

    def __init__(
        self,
        time_window: Optional[float] = None,
        only_with_semantic: bool = True,
    ) -> None:
        self.time_window = time_window
        self.only_with_semantic = only_with_semantic

    def extract(
        self,
        source: MemoryModule,
        target: MemoryModule,
        context: MemoryContext,
    ) -> List[MemoryEntry]:
        # Source must expose get_timeline; if not, fall back to read().
        timeline: List[MemoryEntry]
        if hasattr(source, "get_timeline"):
            timeline = list(source.get_timeline())
        else:
            r = source.read(Query())
            timeline = list(r.entries)

        window = self.time_window
        if window is None:
            ts = getattr(source, "timescales", None)
            window = max(ts) if ts else 600.0

        facts: List[MemoryEntry] = []
        for ev in timeline:
            if self.only_with_semantic and not ev.semantic_keys:
                continue
            subject = ev.semantic_keys[0] if ev.semantic_keys else "something"
            time_anchor = min(ev.temporal_keys) if ev.temporal_keys else 0.0
            fact_text = f"({subject}, was_seen, ~{window:.0f}s around t={time_anchor:.1f})"
            fid = f"fact-{next(_fact_counter)}"
            facts.append(
                MemoryEntry(
                    entry_id=fid,
                    text=fact_text,
                    semantic_keys=[subject, "was_seen"],
                    temporal_keys=[time_anchor],
                    source_slot="episodic",
                    metadata={
                        "kind": "extracted_fact",
                        "subject": subject,
                        "predicate": "was_seen",
                        "time_window": window,
                        "source_entry_id": ev.entry_id,
                    },
                )
            )
        return facts


__all__ = ["ExtractFactsConsolidationPolicy"]
