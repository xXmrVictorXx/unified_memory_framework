"""CLiViS ``TimeWorkingMemory`` — episodic short-term evidence store.

Faithful re-implementation of ``reproduce/CLiViS/clivis/memory/time_working_memory.py``.

Original semantics (per the explore analysis):
* Stores the question, the chat history (OpenAI-format dicts), and a list of
  ``Rationale`` evidence entries.
* Each ``Rationale`` is a structured evidence chunk with free-text body plus
  spatial (``related_area``), temporal (``related_period``), and optional
  object tags.
* The pipeline appends a (LLM instruction, VLM response) pair per round, then
  extracts a new rationale via an LLM call.

This module exposes both the original API (``update_history_msg``,
``extract_and_update_rationale_list``, ``output_memory_info``) and the unimem
ABC API (``write`` / ``read`` / ``get_current`` / ``set_current`` /
``append_event`` / ``get_timeline``).

The LLM extractor is injectable. Tests use ``MockLLM``; production wiring
passes a callable that wraps Qwen / DeepSeek / GPT.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import Query, QueryResult
from unimem.core.slot_abc import EpisodicMemoryABC, WorkingMemoryABC

LLMFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Rationale
# --------------------------------------------------------------------------- #
class Rationale:
    """One piece of evidence linked to a (period, area, object).

    Mirrors ``reproduce/CLiViS/clivis/memory/time_working_memory.py:Rationale``.
    """

    __slots__ = ("evidence", "related_area", "related_period", "related_obj")

    def __init__(
        self,
        evidence: str,
        related_area: str,
        related_period: str,
        related_obj: Optional[str] = None,
    ) -> None:
        self.evidence = evidence
        self.related_area = related_area
        self.related_period = related_period
        self.related_obj = related_obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence": self.evidence,
            "related_area": self.related_area,
            "related_period": self.related_period,
            "related_obj": self.related_obj,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Rationale(area={self.related_area!r}, period={self.related_period!r}, obj={self.related_obj!r})"


# --------------------------------------------------------------------------- #
# TimeWorkingMemory
# --------------------------------------------------------------------------- #
class TimeWorkingMemory(WorkingMemoryABC, EpisodicMemoryABC):
    """CLiViS working memory: question + chat history + rationale list.

    Implementation notes:

    * The unimem ``MemoryEntry`` text is the rationale evidence. ``metadata``
      carries area/period/object so the multi-axis index can retrieve by
      them via ``semantic_keys`` (area/object) and ``temporal_keys`` (period
      parsed to seconds).
    * ``write`` accepts an entry and routes it as a rationale append.
    * The "current" WorkingMemoryABC entry is the last appended rationale
      (or, before any rounds, the question itself as a placeholder).
    """

    timescales = (1.0, 60.0)  # short-term + medium-term buckets

    def __init__(
        self,
        question: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        llm_extractor: Optional[LLMFn] = None,
    ) -> None:
        self.question = question
        self.history_messages: List[Dict[str, str]] = list(messages) if messages else []
        self.rationale_list: List[Rationale] = []
        self._llm_extractor = llm_extractor

    # ------------------------------------------------------------------ #
    # Original CLiViS API
    # ------------------------------------------------------------------ #
    def update_history_msg(self, llm_instruction: str, vlm_response: str) -> None:
        self.history_messages.append({"role": "assistant", "content": llm_instruction})
        self.history_messages.append({"role": "user", "content": vlm_response})

    def get_rationale_count(self) -> int:
        return len(self.rationale_list)

    def extract_and_update_rationale_list(
        self,
        period: str,
        extractor: Optional[LLMFn] = None,
        max_retries: int = 5,
    ) -> Optional[Rationale]:
        """Use the LLM to extract a new rationale from the latest dialogue turn.

        Reproduces ``TimeWorkingMemory.extract_and_update_rationale_list``:
        builds a prompt from the last (instruction, response) pair, asks the
        LLM for an evidence summary, parses it into a :class:`Rationale`.
        Returns the new Rationale (also appended to the list) or None on
        failure / empty extraction.

        The extractor can be passed per-call (production) or set at
        construction time (tests).
        """
        fn = extractor or self._llm_extractor
        if fn is None:
            raise RuntimeError(
                "extract_and_update_rationale_list requires an LLM extractor "
                "(pass llm_extractor to the constructor or this method)"
            )
        if len(self.history_messages) < 2:
            return None
        last_user = self.history_messages[-1]["content"]
        prompt = (
            "Extract a concise piece of evidence from the following VLM response. "
            "Return JSON with keys: evidence, related_area, related_obj (optional). "
            "If no new evidence, return JSON {\"evidence\": \"\"}.\n"
            f"VLM response: {last_user}\n"
            f"Period: {period}\n"
            f"Question: {self.question}"
        )
        raw = fn(prompt)
        parsed = _safe_json_extract(raw)
        if not parsed or not parsed.get("evidence"):
            return None
        rat = Rationale(
            evidence=parsed["evidence"],
            related_area=parsed.get("related_area", ""),
            related_period=period,
            related_obj=parsed.get("related_obj"),
        )
        self.rationale_list.append(rat)
        return rat

    def output_memory_info(self) -> str:
        """Pretty-serialise the entire working memory for LLM context."""
        lines = [f"Question: {self.question}", "Rationales:"]
        for i, r in enumerate(self.rationale_list):
            line = f"  [{i}] area={r.related_area} period={r.related_period}"
            if r.related_obj:
                line += f" obj={r.related_obj}"
            line += f"\n      evidence: {r.evidence}"
            lines.append(line)
        if not self.rationale_list:
            lines.append("  (none yet)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # WorkingMemoryABC
    # ------------------------------------------------------------------ #
    def get_current(self) -> Optional[MemoryEntry]:
        if self.rationale_list:
            r = self.rationale_list[-1]
            return self._rationale_to_entry(r, len(self.rationale_list) - 1)
        if self.question:
            return MemoryEntry(
                entry_id="question",
                text=self.question,
                source_slot="working_memory",
            )
        return None

    def set_current(self, entry: MemoryEntry) -> None:
        # WorkingMemory.set_current in CLiViS just replaces the latest
        # rationale. We accept any entry, but treat its metadata fields
        # as authoritative.
        rat = Rationale(
            evidence=entry.text,
            related_area=entry.metadata.get("related_area", ""),
            related_period=entry.metadata.get("related_period", ""),
            related_obj=entry.metadata.get("related_obj"),
        )
        if self.rationale_list:
            self.rationale_list[-1] = rat
        else:
            self.rationale_list.append(rat)

    # ------------------------------------------------------------------ #
    # EpisodicMemoryABC
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        rat = Rationale(
            evidence=entry.text,
            related_area=entry.metadata.get("related_area", ""),
            related_period=entry.metadata.get("related_period", ""),
            related_obj=entry.metadata.get("related_obj"),
        )
        self.rationale_list.append(rat)

    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        # Periods are strings in CLiViS; here we approximate by parsing
        # seconds from the period string and filter on it.
        out: List[MemoryEntry] = []
        for i, r in enumerate(self.rationale_list):
            t = _period_to_seconds(r.related_period)
            if t is None:
                continue
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t > t_max:
                continue
            out.append(self._rationale_to_entry(r, i))
        return out

    # ------------------------------------------------------------------ #
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        self.append_event(entry)
        return True

    def read(self, query: Query) -> QueryResult:
        sem = set(query.semantic or [])
        entries = []
        for i, r in enumerate(self.rationale_list):
            entry = self._rationale_to_entry(r, i)
            if sem:
                keys = set(entry.semantic_keys)
                # AND semantics within axis: every queried key must be present
                if not sem.issubset(keys):
                    continue
            entries.append(entry)
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot="working_memory")

    def clear(self) -> None:
        self.rationale_list.clear()
        self.history_messages.clear()
        self.question = ""

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.rationale_list),
            "n_messages": len(self.history_messages),
            "has_question": bool(self.question),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _rationale_to_entry(self, r: Rationale, index: int) -> MemoryEntry:
        semantic = []
        if r.related_area:
            semantic.append(r.related_area)
        if r.related_obj:
            semantic.append(r.related_obj)
        temporal = []
        t = _period_to_seconds(r.related_period)
        if t is not None:
            temporal.append(t)
        return MemoryEntry(
            entry_id=f"rationale-{index}",
            text=r.evidence,
            semantic_keys=semantic,
            temporal_keys=temporal,
            source_slot="working_memory",
            metadata={
                "related_area": r.related_area,
                "related_period": r.related_period,
                "related_obj": r.related_obj,
                "rationale_index": index,
            },
        )


# --------------------------------------------------------------------------- #
# Period-string parsing helpers (lifted from CLiViS utils)
# --------------------------------------------------------------------------- #
def _period_to_seconds(period: str) -> Optional[float]:
    """Parse ``"hh:mm:ss - hh:mm:ss"`` to a float midpoint in seconds."""
    if not period:
        return None
    # Find all hh:mm:ss patterns
    matches = re.findall(r"(\d+):(\d+):(\d+)", period)
    if not matches:
        return None
    seconds = [int(h) * 3600 + int(m) * 60 + int(s) for (h, m, s) in matches]
    return sum(seconds) / len(seconds)


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Try to recover a JSON object from an LLM response.

    LLMs often wrap JSON in ```json``` fences or trailing prose. We try:
    direct json.loads, then the first {...} block, then JSON from a code fence.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try first {...} block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Try fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


__all__ = ["Rationale", "TimeWorkingMemory", "_period_to_seconds", "_safe_json_extract"]
