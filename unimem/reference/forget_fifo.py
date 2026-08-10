"""``FIFOForgetPolicy`` — reference forget policy.

Drops the globally-oldest entries from a :class:`ListEpisodicMemory` (or any
module exposing a ``drop_oldest(n)`` method) whenever its size exceeds a cap.
Models simple capacity management — the kind of forgetting every paper in the
survey does in some form.
"""
from __future__ import annotations

from typing import Any

from ..core.context import MemoryContext
from ..core.module import MemoryModule
from ..policies.forget_policy import ForgetPolicy


class FIFOForgetPolicy(ForgetPolicy):
    """First-In-First-Out forgetting with a hard ``capacity`` ceiling.

    Compatible with any module whose ``stats()`` returns a ``count`` field
    and which exposes a ``drop_oldest(n)`` method (as
    :class:`~unimem.reference.episodic_memory.ListEpisodicMemory` does).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = int(capacity)

    def apply(self, module: MemoryModule, context: MemoryContext) -> int:
        stats = module.stats()
        current = stats.get("count", 0)
        if not isinstance(current, (int, float)) or current <= self.capacity:
            return 0
        surplus = int(current) - self.capacity
        drop_fn = getattr(module, "drop_oldest", None)
        if drop_fn is None:
            return 0
        return int(drop_fn(surplus))


__all__ = ["FIFOForgetPolicy"]
