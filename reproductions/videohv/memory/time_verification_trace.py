"""``VerificationTraceMemory`` — storage-backed short-term episodic log.

A thin facade over :class:`~unimem.graph_storage.base.GraphStorage` that
preserves VideoHV-Agent's high-level API (``record_round`` / ``get_round``
/ ``all_rounds``) while persisting each round as a node labelled
``:episodic:Trace`` with an attached ``:TimeIndex`` node carrying the
``round_index``.

Inherits from :class:`~unimem.core.module.MemoryModule` so the
:class:`~unimem.graph.graph.MemoryGraph` can host it as a node.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage
from unimem.graph_storage.time_index import attach_timestamp

TRACE_LABEL = "Trace"


class VerificationTraceMemory(MemoryModule):
    """Per-refinement-round trace: hypotheses, clue, verdict, distinction score.

    Each round is stored as a graph node ``:episodic:Trace`` keyed by
    ``trace-round-<idx>`` with a ``:TimeIndex`` attached via ``:AT_TIME``
    (carrying ``timestamp=round_index``).
    """

    SLOT = MemorySlot.EM

    def __init__(self, graph_storage: Optional[GraphStorage] = None) -> None:
        super().__init__(slot=self.SLOT)
        self._gs = graph_storage or InMemoryGraphStorage()
        # round_index → dict of fields (legacy in-memory mirror)
        self._rounds: List[Dict[str, Any]] = []

    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

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
        # Mirror to storage.
        self._persist_round(round_dict)

    def _persist_round(self, round_dict: Dict[str, Any]) -> None:
        round_idx = round_dict.get("round", 0)
        node_id = f"trace-round-{round_idx}"
        text_parts: List[str] = [f"Round {round_idx}"]
        sem_keys: List[str] = [f"round-{round_idx}"]
        if "clue" in round_dict:
            text_parts.append(f"clue: {round_dict['clue']}")
            sem_keys.append("clue")
        if "verdict" in round_dict:
            text_parts.append(f"verdict: {round_dict['verdict']}")
            sem_keys.append("verdict")
            sem_keys.append(str(round_dict["verdict"]).lower())
        if "hypotheses" in round_dict:
            text_parts.append(f"hypotheses: {round_dict['hypotheses']}")
            sem_keys.append("hypothesis")
        if "answer_choice" in round_dict:
            text_parts.append(f"answer: option-{round_dict['answer_choice']}")
        self._gs.add_node(
            node_id,
            [self.SLOT.value, TRACE_LABEL],
            {
                "text": " | ".join(text_parts),
                "semantic_keys": sem_keys,
                "temporal_keys": [float(round_idx)],
                "metadata": dict(round_dict),
                "source_slot": self.SLOT.value,
                "round": round_idx,
            },
        )
        attach_timestamp(self._gs, node_id, float(round_idx))

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
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        round_idx = entry.metadata.get("round", len(self._rounds))
        self.record_round(
            round_index=round_idx,
            hypotheses=entry.metadata.get("hypotheses"),
            clue=entry.metadata.get("clue"),
            distinction_score=entry.metadata.get("distinction_score"),
            verdict=entry.metadata.get("verdict"),
            answer_choice=entry.metadata.get("answer_choice"),
        )

    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        self.append_event(entry)
        return True

    def read(self, query: Query) -> QueryResult:
        sem = set(query.semantic or [])
        entries: List[MemoryEntry] = []
        for r in self._rounds:
            entry = self._round_to_entry(r)
            if sem and not sem.issubset(set(entry.semantic_keys)):
                continue
            entries.append(entry)
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot=self.SLOT.value)

    def clear(self) -> None:
        self._rounds.clear()
        self._gs.query(
            f"MATCH (n:{self.SLOT.value}:{TRACE_LABEL}) DETACH DELETE n"
        )

    def stats(self) -> Dict[str, Any]:
        n_verdicts = sum(1 for r in self._rounds if "verdict" in r)
        return {
            "count": len(self._rounds),
            "n_with_verdict": n_verdicts,
        }

    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        rounds: List[MemoryEntry] = []
        for r in self._rounds:
            t = float(r["round"])
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t > t_max:
                continue
            rounds.append(self._round_to_entry(r))
        return rounds

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _round_to_entry(self, round_dict: Dict[str, Any]) -> MemoryEntry:
        round_idx = round_dict.get("round", 0)
        text_parts: List[str] = [f"Round {round_idx}"]
        sem_keys: List[str] = [f"round-{round_idx}"]
        if "clue" in round_dict:
            text_parts.append(f"clue: {round_dict['clue']}")
            sem_keys.append("clue")
        if "verdict" in round_dict:
            text_parts.append(f"verdict: {round_dict['verdict']}")
            sem_keys.append("verdict")
            sem_keys.append(str(round_dict["verdict"]).lower())
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
            source_slot=self.SLOT.value,
            metadata=dict(round_dict),
        )


__all__ = ["VerificationTraceMemory", "TRACE_LABEL"]
