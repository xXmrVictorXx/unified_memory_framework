"""Registry + factory façade for plug-in module construction.

The :class:`Registry` is the single source of truth for "which implementations
are available for which slot". A downstream EQA method plugs its custom module
in by registering it once with ``@registry.register_module_decorator(slot,
name)``; from then on the declarative :class:`GraphSpec` / dict pipeline can
mention it by name.
"""
from __future__ import annotations

from .memory_factory import MemoryFactory
from .registry import Registry

__all__ = ["Registry", "MemoryFactory"]
