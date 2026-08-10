"""Forget / capacity-management policies.

A ``ForgetPolicy`` runs at the end of each consolidation pass and asks each
module to drop entries it no longer wants to keep. Implementations include:

* ``NoOp`` — never forget (the default).
* The reference ``FIFOForgetPolicy`` lives in :mod:`unimem.reference`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.context import MemoryContext
from ..core.module import MemoryModule


class ForgetPolicy(ABC):
    """Tell ``module`` to drop unwanted entries."""

    @abstractmethod
    def apply(self, module: MemoryModule, context: MemoryContext) -> int:
        """Drop entries. Return the number of entries actually removed."""


class NoOp(ForgetPolicy):
    """Never forget anything."""

    def apply(self, module: MemoryModule, context: MemoryContext) -> int:
        return 0


__all__ = ["ForgetPolicy", "NoOp"]
