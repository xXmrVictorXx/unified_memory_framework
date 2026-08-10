"""Edge kinds + the edge dataclass.

Edge kinds are kept deliberately small (5) — enough to express every memory
fusion pattern from the survey, no more. The optional ``policy`` slot carries
a strategy on edges that need one:

* ``FEEDS``            → ``WritePolicy`` (event-triggered gating)
* ``CONSOLIDATES_TO``  → ``ConsolidationPolicy``
* ``INDEXES``/``REFERENCES``/``SUBSUMES`` → unused (carry metadata only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EdgeKind(Enum):
    FEEDS = "feeds"
    CONSOLIDATES_TO = "consolidates_to"
    INDEXES = "indexes"
    REFERENCES = "references"
    SUBSUMES = "subsumes"

    @classmethod
    def from_value(cls, value: "EdgeKind | str") -> "EdgeKind":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise KeyError(f"Unknown EdgeKind value: {value!r}")


@dataclass
class MemoryEdge:
    """A directed edge ``source_id -> target_id`` of a given ``kind``."""

    source_id: str
    target_id: str
    kind: EdgeKind
    policy: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow string-typed kind for ergonomic dict-based construction.
        if not isinstance(self.kind, EdgeKind):
            self.kind = EdgeKind.from_value(self.kind)
        if self.source_id == self.target_id:
            raise ValueError(
                f"Self-loop not allowed: source_id==target_id=={self.source_id!r}"
            )


__all__ = ["EdgeKind", "MemoryEdge"]
