"""Write-side gating policies.

A ``WritePolicy`` decides whether a (module, entry, context) tuple should be
admitted. Edge-level policies model INHerit-SG-style *event-triggered writes*
(only ingest on topological change); module-level policies model per-memory
concerns (e.g. a WM that always overwrites vs. an SM that only adds novel
facts).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry
from ..core.module import MemoryModule


class WritePolicy(ABC):
    """Decide whether ``module`` should store ``entry`` at ``context``."""

    @abstractmethod
    def should_write(
        self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext
    ) -> bool:
        """Return True to admit, False to skip (and stop graph propagation)."""


class AlwaysWrite(WritePolicy):
    """Admit everything. Default for FEEDS edges and root nodes."""

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return True


class NeverWrite(WritePolicy):
    """Admit nothing. Useful for stub / read-only modules."""

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return False


class LambdaWritePolicy(WritePolicy):
    """Wrap a plain callable as a :class:`WritePolicy` (handy in tests/scripts)."""

    def __init__(self, fn: Any) -> None:
        if not callable(fn):
            raise TypeError("LambdaWritePolicy expects a callable")
        self._fn = fn

    def should_write(self, module: MemoryModule, entry: MemoryEntry, context: MemoryContext) -> bool:
        return bool(self._fn(module, entry, context))


__all__ = ["WritePolicy", "AlwaysWrite", "NeverWrite", "LambdaWritePolicy"]
