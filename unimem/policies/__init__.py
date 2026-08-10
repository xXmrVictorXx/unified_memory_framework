"""Cross-cutting policies: write gating, read merging, consolidation, forgetting.

Policies are *strategies* that can be plugged in at three levels:

1. **Per-module** via the ``write_policy`` / ``read_policy`` / ``forget_policy``
   attributes on :class:`~unimem.core.module.MemoryModule`.
2. **Per-edge** on FEEDS (``WritePolicy``) and CONSOLIDATES_TO
   (``ConsolidationPolicy``) edges in :class:`~unimem.graph.graph.MemoryGraph`.
3. **Graph-level defaults** used when neither of the above is set.

Every policy is a small ABC with one method. Defaults (``AlwaysWrite``,
``ConcatRead``, ``Passthrough``, ``NoOp``) are intentionally trivial so they
can serve as identity baselines in tests and ablations.
"""
from __future__ import annotations

from .consolidation_policy import (
    ConsolidationPolicy,
    Passthrough,
)
from .forget_policy import ForgetPolicy, NoOp
from .read_policy import ConcatRead, ReadPolicy
from .write_policy import AlwaysWrite, NeverWrite, WritePolicy

__all__ = [
    "WritePolicy",
    "AlwaysWrite",
    "NeverWrite",
    "ReadPolicy",
    "ConcatRead",
    "ConsolidationPolicy",
    "Passthrough",
    "ForgetPolicy",
    "NoOp",
]
