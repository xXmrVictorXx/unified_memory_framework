"""``MemoryNode`` wraps a :class:`MemoryModule` and gives it graph identity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Union

from ..core.module import MemoryModule
from ..core.slots import MemorySlot


@dataclass
class MemoryNode:
    """A vertex in the memory graph: id + slot + module + bookkeeping."""

    node_id: str
    slot: MemorySlot
    module: MemoryModule
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept string slots for ergonomic dict-based construction.
        if not isinstance(self.slot, MemorySlot):
            self.slot = MemorySlot.from_value(self.slot)
        if not self.label:
            self.label = f"{self.slot.name}:{self.node_id}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MemoryNode({self.node_id!r}, slot={self.slot.name})"


__all__ = ["MemoryNode"]
