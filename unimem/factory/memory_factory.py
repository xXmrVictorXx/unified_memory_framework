"""``MemoryFactory`` — thin convenience façade over :class:`Registry`.

The registry already does the heavy lifting; this façade just adds a few
shortcut constructors that pre-fill common slots/policies. Most user code
will interact with the registry directly via the builder.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from ..core.module import MemoryModule
from ..core.slots import MemorySlot
from .registry import Registry


class MemoryFactory:
    """Convenience wrapper around a :class:`Registry`."""

    def __init__(self, registry: Optional[Registry] = None) -> None:
        self.registry = registry or Registry()

    # -- module construction --------------------------------------------- #
    def make_module(
        self,
        slot: Union[MemorySlot, str],
        impl_name: str,
        **kwargs: Any,
    ) -> MemoryModule:
        return self.registry.create_module(slot, impl_name, **kwargs)

    def make_policy(
        self,
        policy_type: str,
        name: str,
        **kwargs: Any,
    ) -> Any:
        return self.registry.create_policy(policy_type, name, **kwargs)

    # -- registration pass-throughs ------------------------------------- #
    def register_module(self, slot, name, cls):
        self.registry.register_module(slot, name, cls)
        return self

    def register_module_decorator(self, slot, name):
        return self.registry.register_module_decorator(slot, name)

    def register_policy(self, policy_type, name, cls):
        self.registry.register_policy(policy_type, name, cls)
        return self


__all__ = ["MemoryFactory"]
