"""CLiViS ``TimeWorkingMemory`` — storage-backed episodic short-term store.

A thin facade over :class:`~unimem.graph_storage.base.GraphStorage` that
preserves CLiViS's high-level API (question, chat history, rationale list)
without subclassing any unimem slot ABC. It still inherits from
:class:`~unimem.core.module.MemoryModule` so the
:class:`~unimem.graph.graph.MemoryGraph` can host it as a node.

Each rationale is stored as a node ``:working_memory:Rationale`` with a
``:TimeIndex`` node attached via ``:AT_TIME`` (period parsed to seconds).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage
from unimem.graph_storage.time_index import attach_period

LLMFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Rationale
# --------------------------------------------------------------------------- #
class Rationale:
    """One piece of evidence linked to a (period, area, object)."""

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
        return (
            f"Rationale(area={self.related_area!r}, "
            f"period={self.related_period!r}, obj={self.related_obj!r})"
        )


class TimeWorkingMemory(MemoryModule):
    """CLiViS working memory: question + chat history + rationale list.

    Rationales are persisted to the bound :class:`GraphStorage` so the
    pipeline can introspect them via Cypher queries if desired. The legacy
    in-memory attributes (``rationale_list`` / ``history_messages``) are
    preserved for compatibility with the existing pipeline code.
    """

    SLOT = MemorySlot.WM

    def __init__(
        self,
        question: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        llm_extractor: Optional[LLMFn] = None,
        graph_storage: Optional[GraphStorage] = None,
    ) -> None:
        super().__init__(slot=self.SLOT)
        self.question = question
        self.history_messages: List[Dict[str, str]] = list(messages) if messages else []
        self.rationale_list: List[Rationale] = []
        self._llm_extractor = llm_extractor
        self._gs = graph_storage or InMemoryGraphStorage()

    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

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
        self._append_rationale(rat)
        return rat

    def output_memory_info(self) -> str:
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
    # Storage integration
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        """Append a rationale from a MemoryEntry (legacy episodic API)."""
        self._append_rationale(Rationale(
            evidence=entry.text,
            related_area=entry.metadata.get("related_area", ""),
            related_period=entry.metadata.get("related_period", ""),
            related_obj=entry.metadata.get("related_obj"),
        ))

    def _append_rationale(self, rat: Rationale) -> None:
        self.rationale_list.append(rat)
        idx = len(self.rationale_list) - 1
        node_id = f"rationale-{idx}"
        semantic = []
        if rat.related_area:
            semantic.append(rat.related_area)
        if rat.related_obj:
            semantic.append(rat.related_obj)
        self._gs.add_node(
            node_id,
            [self.SLOT.value, "Rationale"],
            {
                "text": rat.evidence,
                "evidence": rat.evidence,
                "related_area": rat.related_area,
                "related_period": rat.related_period,
                "related_obj": rat.related_obj or "",
                "rationale_index": idx,
            },
        )
        if rat.related_period:
            # Parse start/end seconds from the period string so that the
            # TimeIndex node can be queried via time_range / start_t / end_t.
            from .navigation_graph import _parse_range_seconds
            rng = _parse_range_seconds(rat.related_period)
            if rng is not None:
                attach_period(
                    self._gs,
                    node_id,
                    rat.related_period,
                    start_t=rng[0],
                    end_t=rng[1],
                )
            else:
                attach_period(self._gs, node_id, rat.related_period)

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.rationale_list),
            "n_messages": len(self.history_messages),
            "has_question": bool(self.question),
        }

    def clear(self) -> None:
        self.rationale_list.clear()
        self.history_messages.clear()
        self.question = ""
        self._gs.query(
            f"MATCH (n:{self.SLOT.value}) DETACH DELETE n"
        )

    # ------------------------------------------------------------------ #
    # MemoryModule contract — append rationale from entry
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        self._append_rationale(Rationale(
            evidence=entry.text,
            related_area=entry.metadata.get("related_area", ""),
            related_period=entry.metadata.get("related_period", ""),
            related_obj=entry.metadata.get("related_obj"),
        ))
        return True

    def read(self, query: Query) -> QueryResult:
        sem = set(query.semantic or [])
        entries: List[MemoryEntry] = []
        for i, r in enumerate(self.rationale_list):
            entry = self._rationale_to_entry(r, i)
            if sem and not sem.issubset(set(entry.semantic_keys)):
                continue
            entries.append(entry)
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot=self.SLOT.value)

    def _rationale_to_entry(self, r: Rationale, index: int) -> MemoryEntry:
        semantic = []
        if r.related_area:
            semantic.append(r.related_area)
        if r.related_obj:
            semantic.append(r.related_obj)
        return MemoryEntry(
            entry_id=f"rationale-{index}",
            text=r.evidence,
            semantic_keys=semantic,
            source_slot=self.SLOT.value,
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
    matches = re.findall(r"(\d+):(\d+):(\d+)", period)
    if not matches:
        return None
    seconds = [int(h) * 3600 + int(m) * 60 + int(s) for (h, m, s) in matches]
    return sum(seconds) / len(seconds)


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON object extraction from LLM output (see pipeline._safe_json_extract)."""
    if not text:
        return None
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            candidate = fence.group(1)
    balanced = _extract_balanced(candidate, "{", "}")
    if balanced is not None:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    return None


def _extract_balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the first top-level balanced ``open_ch ... close_ch`` substring.
    See pipeline._extract_balanced for details."""
    pos = 0
    while True:
        start = s.find(open_ch, pos)
        if start < 0:
            return None
        prefix = s[:start]
        if prefix.count("[") > prefix.count("]"):
            pos = start + 1
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        pos = start + 1


__all__ = [
    "Rationale",
    "TimeWorkingMemory",
    "_period_to_seconds",
    "_safe_json_extract",
]
