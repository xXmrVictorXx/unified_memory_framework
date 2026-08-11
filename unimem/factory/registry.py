"""``Registry`` — single flat registry keyed by ``(slot, impl_name)``.

A single registry (rather than 6 per-slot ones) is simpler to maintain and
introspect, and supports ``list_implementations(slot)`` queries naturally.

The registry also stores policies by ``(policy_kind, name)`` so the builder
can reference them by name in dict specs.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ..core.module import MemoryModule
from ..core.slots import MemorySlot
from ..policies.consolidation_policy import ConsolidationPolicy
from ..policies.forget_policy import ForgetPolicy
from ..policies.read_policy import ReadPolicy
from ..policies.write_policy import WritePolicy

# Policy kinds we know how to register.
_POLICY_TYPES: Dict[str, Type[Any]] = {
    "write": WritePolicy,
    "read": ReadPolicy,
    "consolidation": ConsolidationPolicy,
    "forget": ForgetPolicy,
}


class Registry:
    """Maps ``(slot, impl_name) -> module class`` and policy names -> classes."""

    def __init__(self) -> None:
        self._modules: Dict[Tuple[MemorySlot, str], Type[MemoryModule]] = {}
        self._policies: Dict[Tuple[str, str], Type[Any]] = {}

    # ------------------------------------------------------------------ #
    # Module registration
    # ------------------------------------------------------------------ #
    def register_module(
        self,
        slot: Union[MemorySlot, str],
        impl_name: str,
        cls: Type[MemoryModule],
    ) -> None:
        """Register ``cls`` as the implementation ``impl_name`` for ``slot``.

        Re-registering the same ``(slot, impl_name)`` overwrites silently —
        this is convenient for hot-reload during development.
        """
        if not isinstance(cls, type):
            raise TypeError(f"register_module expected a class, got {type(cls)!r}")
        if not issubclass(cls, MemoryModule):
            raise TypeError(
                f"register_module expected subclass of MemoryModule, got {cls!r}"
            )
        slot_member = MemorySlot.from_value(slot)
        key = (slot_member, impl_name)
        self._modules[key] = cls

    def register_module_decorator(
        self, slot: Union[MemorySlot, str], impl_name: str
    ) -> Callable[[Type[MemoryModule]], Type[MemoryModule]]:
        """Decorator form of :meth:`register_module`.

        Usage::

            @registry.register_module_decorator(MemorySlot.EM, "list")
            class ListEpisodicMemory(EpisodicMemoryABC): ...
        """

        def _decorator(cls: Type[MemoryModule]) -> Type[MemoryModule]:
            self.register_module(slot, impl_name, cls)
            return cls

        return _decorator

    def create_module(
        self,
        slot: Union[MemorySlot, str],
        impl_name: str,
        **kwargs: Any,
    ) -> MemoryModule:
        """Instantiate the ``(slot, impl_name)`` module with ``kwargs``.

        The ``slot`` argument is auto-injected into ``kwargs`` (as the slot
        enum) when (a) the caller did not supply it and (b) the registered
        class's ``__init__`` accepts it. This means a registered module
        class that uses the default :class:`MemoryModule` constructor
        signature can be created without repeating the slot explicitly,
        while legacy classes with their own ``__init__`` (e.g.
        ``FakeEpisodic(capacity=...)``) continue to work unchanged.
        """
        import inspect

        slot_member = MemorySlot.from_value(slot)
        cls = self._modules.get((slot_member, impl_name))
        if cls is None:
            raise KeyError(
                f"No module registered for ({slot_member.name!r}, {impl_name!r}). "
                f"Available: {self.list_implementations(slot_member)}"
            )
        if "slot" not in kwargs:
            try:
                sig = inspect.signature(cls.__init__)
                if "slot" in sig.parameters:
                    kwargs = {**kwargs, "slot": slot_member}
            except (TypeError, ValueError):
                pass
        return cls(**kwargs)

    def is_registered(
        self, slot: Union[MemorySlot, str], impl_name: str
    ) -> bool:
        slot_member = MemorySlot.from_value(slot)
        return (slot_member, impl_name) in self._modules

    # ------------------------------------------------------------------ #
    # Policy registration
    # ------------------------------------------------------------------ #
    def register_policy(
        self,
        policy_type: str,
        name: str,
        cls: Type[Any],
    ) -> None:
        """Register a policy class under ``(policy_type, name)``.

        ``policy_type`` must be one of ``"write"``, ``"read"``,
        ``"consolidation"``, ``"forget"``.
        """
        if policy_type not in _POLICY_TYPES:
            raise ValueError(
                f"Unknown policy_type {policy_type!r}. "
                f"Expected one of {sorted(_POLICY_TYPES)}"
            )
        expected = _POLICY_TYPES[policy_type]
        if not (isinstance(cls, type) and issubclass(cls, expected)):
            raise TypeError(
                f"Policy for {policy_type!r} must subclass {expected.__name__}, got {cls!r}"
            )
        self._policies[(policy_type, name)] = cls

    def create_policy(
        self,
        policy_type: str,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a registered policy."""
        cls = self._policies.get((policy_type, name))
        if cls is None:
            available = sorted(
                n for (pt, n) in self._policies if pt == policy_type
            )
            raise KeyError(
                f"No {policy_type!r} policy named {name!r}. Available: {available}"
            )
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def list_implementations(
        self, slot: Optional[Union[MemorySlot, str]] = None
    ) -> Dict[str, List[str]]:
        """Return ``{slot_name: [impl_name, ...]}``.

        With no ``slot`` argument, returns implementations for every slot.
        """
        if slot is None:
            out: Dict[str, List[str]] = {s.name: [] for s in MemorySlot}
            for (s, n) in self._modules:
                out[s.name].append(n)
            for k in out:
                out[k].sort()
            return out
        s = MemorySlot.from_value(slot)
        return {
            s.name: sorted(n for (ss, n) in self._modules if ss == s)
        }

    def list_policies(self, policy_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Return ``{policy_type: [name, ...]}``."""
        out: Dict[str, List[str]] = {k: [] for k in _POLICY_TYPES}
        for (pt, n) in self._policies:
            out[pt].append(n)
        for k in out:
            out[k].sort()
        if policy_type is not None:
            if policy_type not in out:
                raise ValueError(f"Unknown policy_type {policy_type!r}")
            return {policy_type: out[policy_type]}
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Registry(modules={sum(len(v) for v in self.list_implementations().values())}, "
            f"policies={sum(len(v) for v in self.list_policies().values())})"
        )


__all__ = ["Registry"]
