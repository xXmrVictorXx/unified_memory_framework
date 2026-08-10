"""``VerificationTraceMemory`` — short-term episodic log of the verification loop.

VideoHV-Agent's runner.py keeps two inter-round variables locally:

* ``verification_trace_text`` (str) — accumulated verification verdict
* ``prior_hypothesis_lines`` (list[str]) — previous-round hypothesis text

These are short-term episodic state — exactly what unimem's
:class:`~unimem.core.slot_abc.EpisodicMemoryABC` is for. Treating them as a
real (readable) memory makes the agent loop introspectable: a critic agent
could query "what verdicts did we already form?" through the standard
``MemoryModule.read`` interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import Query, QueryResult
from unimem.core.slot_abc import EpisodicMemoryABC


class VerificationTraceMemory(EpisodicMemoryABC):
    """Per-refinement-round trace: hypotheses, clue, verdict, distinction score."""

    timescales = (1.0,)  # all short-term

    def __init__(self) -> None:
        # round_index → dict of fields
        self._rounds: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Original-style mutators
    # ------------------------------------------------------------------ #
    def record_round(
        self,
        round_index: int,
        hypotheses: Optional[List[str]] = None,
        clue: Optional[str] = None,
        distinction_score: Optional[float] = None,
        verdict: Optional[str] = None,
        answer_choice: Optional[int] = None,
    ) -> None:
        """Append (or update) the trace for a given refinement round."""
        # Ensure list is long enough
        while len(self._rounds) <= round_index:
            self._rounds.append({"round": len(self._rounds)})
        round_dict = self._rounds[round_index]
        if hypotheses is not None:
            round_dict["hypotheses"] = list(hypotheses)
        if clue is not None:
            round_dict["clue"] = clue
        if distinction_score is not None:
            round_dict["distinction_score"] = float(distinction_score)
        if verdict is not None:
            round_dict["verdict"] = verdict
        if answer_choice is not None:
            round_dict["answer_choice"] = int(answer_choice)

    @property
    def n_rounds(self) -> int:
        return len(self._rounds)

    def get_round(self, round_index: int) -> Optional[Dict[str, Any]]:
        if 0 <= round_index < len(self._rounds):
            return dict(self._rounds[round_index])
        return None

    def all_rounds(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._rounds]

    # ------------------------------------------------------------------ #
    # EpisodicMemoryABC
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        # Treat entry as a single round's record encoded via metadata
        round_idx = entry.metadata.get("round", len(self._rounds))
        self.record_round(
            round_index=round_idx,
            hypotheses=entry.metadata.get("hypotheses"),
            clue=entry.metadata.get("clue"),
            distinction_score=entry.metadata.get("distinction_score"),
            verdict=entry.metadata.get("verdict"),
            answer_choice=entry.metadata.get("answer_choice"),
        )

    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        # We treat round_index as the time axis (each round = one "tick")
        rounds = []
        for r in self._rounds:
            t = float(r["round"])
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t > t_max:
                continue
            rounds.append(self._round_to_entry(r))
        return rounds

    # ------------------------------------------------------------------ #
    # MemoryModule
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        self.append_event(entry)
        return True

    def read(self, query: Query) -> QueryResult:
        sem = set(query.semantic or [])
        entries: List[MemoryEntry] = []
        for r in self._rounds:
            entry = self._round_to_entry(r)
            if sem:
                if not sem.issubset(set(entry.semantic_keys)):
                    continue
            entries.append(entry)
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot="episodic")

    def clear(self) -> None:
        self._rounds.clear()

    def stats(self) -> Dict[str, Any]:
        n_verdicts = sum(1 for r in self._rounds if "verdict" in r)
        return {
            "count": len(self._rounds),
            "n_with_verdict": n_verdicts,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _round_to_entry(self, round_dict: Dict[str, Any]) -> MemoryEntry:
        round_idx = round_dict.get("round", 0)
        text_parts = [f"Round {round_idx}"]
        sem_keys = [f"round-{round_idx}"]
        if "clue" in round_dict:
            text_parts.append(f"clue: {round_dict['clue']}")
            sem_keys.append("clue")
        if "verdict" in round_dict:
            text_parts.append(f"verdict: {round_dict['verdict']}")
            sem_keys.append("verdict")
            sem_keys.append(round_dict["verdict"].lower())
        if "hypotheses" in round_dict:
            text_parts.append(f"hypotheses: {round_dict['hypotheses']}")
            sem_keys.append("hypothesis")
        if "answer_choice" in round_dict:
            text_parts.append(f"answer: option-{round_dict['answer_choice']}")
        return MemoryEntry(
            entry_id=f"trace-round-{round_idx}",
            text=" | ".join(text_parts),
            semantic_keys=sem_keys,
            temporal_keys=[float(round_idx)],
            source_slot="episodic",
            metadata=dict(round_dict),
        )


__all__ = ["VerificationTraceMemory"]
