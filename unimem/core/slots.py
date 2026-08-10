"""The six converged memory slots.

These slots come from a survey of ~55 EQA / embodied-agent papers: every
memory module observed fits into one of the six functional roles below.
A system may leave any slot unpopulated; no paper in the survey filled all
six. The framework treats slots as *roles*, not implementations.
"""
from __future__ import annotations

from enum import Enum


class MemorySlot(Enum):
    """The six functional memory slots.

    Members, in rough order from shortest-lived to longest-lived:

    * ``WM`` — working memory: current observation, task state, recent context.
    * ``SG`` — scene graph: hierarchical object-relation topology.
    * ``GM`` — spatial/geometric memory: metric map, occupancy, navigability.
    * ``EM`` — episodic memory: time-ordered events / observation sequences.
    * ``SM`` — semantic / knowledge memory: facts, rules, common sense.
    * ``PM`` — procedural / skill memory: action policies, capability profiles.
    """

    WM = "working_memory"
    SG = "scene_graph"
    GM = "spatial_geometric"
    EM = "episodic"
    SM = "semantic"
    PM = "procedural"

    @classmethod
    def from_value(cls, value: str) -> "MemorySlot":
        """Look up a slot by its string value (case-insensitive)."""
        if isinstance(value, MemorySlot):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise KeyError(f"Unknown MemorySlot value: {value!r}")


__all__ = ["MemorySlot"]
