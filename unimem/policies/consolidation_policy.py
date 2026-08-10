"""Consolidation policies.

A ``ConsolidationPolicy`` extracts "sedimentable" entries from a source module
to be written into a target module along a CONSOLIDATES_TO edge. This is the
edge-level equivalent of :meth:`MemoryModule.consolidate`.

Edge-level policies are preferred because they let the *same* source module
consolidate differently into different targets (e.g. EM → SM extracts facts,
EM → GM extracts spatial patterns).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.context import MemoryContext
from ..core.entry import MemoryEntry
from ..core.module import MemoryModule


class ConsolidationPolicy(ABC):
    """Extract entries from ``source`` to be ingested by ``target``."""

    @abstractmethod
    def extract(
        self,
        source: MemoryModule,
        target: MemoryModule,
        context: MemoryContext,
    ) -> List[MemoryEntry]:
        """Return the list of entries to write into ``target``.

        Returning an empty list is normal — many passes will produce nothing.
        """


class Passthrough(ConsolidationPolicy):
    """Identity consolidation: delegates to ``source.consolidate(target, ...)``.

    Use this when the source module already implements ``consolidate`` itself
    and you just want the edge to wire it up.
    """

    def extract(
        self,
        source: MemoryModule,
        target: MemoryModule,
        context: MemoryContext,
    ) -> List[MemoryEntry]:
        return list(source.consolidate(target, context))


__all__ = ["ConsolidationPolicy", "Passthrough"]
