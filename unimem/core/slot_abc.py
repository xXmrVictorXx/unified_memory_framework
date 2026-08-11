"""Slot-specialised ABCs (optional mixins).

Each slot ABC extends :class:`MemoryModule` (and ``ABC``) and adds only 2–3
slot-specific abstract methods on top. A concrete implementation picks
**one** slot ABC to inherit from to gain both the storage-backed
``write`` / ``read`` / ``clear`` / ``stats`` defaults and the slot-specific
convenience method contract (e.g. ``add_object`` for a scene graph).

These mixins are **optional** — modules that only need the storage-backed
defaults can subclass :class:`MemoryModule` directly without inheriting a
slot ABC. The slot ABCs exist for the (common) case where you want the
framework to enforce that a scene-graph module really implements
``get_children`` etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .entry import MemoryEntry
from .module import MemoryModule


# --------------------------------------------------------------------------- #
# WM — working memory
# --------------------------------------------------------------------------- #
class WorkingMemoryABC(MemoryModule, ABC):
    """Short-lived scratchpad: current observation, task state, recent context."""

    @abstractmethod
    def get_current(self) -> Optional[MemoryEntry]:
        """Return the current / most recent entry, or ``None`` if empty."""

    @abstractmethod
    def set_current(self, entry: MemoryEntry) -> None:
        """Overwrite the current entry."""


# --------------------------------------------------------------------------- #
# SG — scene graph
# --------------------------------------------------------------------------- #
class SceneGraphMemoryABC(MemoryModule, ABC):
    """Hierarchical object-relation topology (floor → room → object → ...)."""

    @abstractmethod
    def add_object(
        self, object_id: str, parent_id: Optional[str] = None, **attrs: Any
    ) -> bool:
        """Insert an object (optionally under a parent). Returns success."""

    @abstractmethod
    def get_children(self, parent_id: Optional[str]) -> List[str]:
        """Return ids of the direct children of ``parent_id``.

        ``parent_id=None`` returns the roots of the forest.
        """

    @abstractmethod
    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Return the attribute dict for ``object_id`` or ``None``."""


# --------------------------------------------------------------------------- #
# GM — spatial / geometric
# --------------------------------------------------------------------------- #
class SpatialGeometricMemoryABC(MemoryModule, ABC):
    """Metric / topological map: occupancy, regions, navigability."""

    @abstractmethod
    def is_navigable(self, point: Sequence[float]) -> bool:
        """Whether the robot can reach ``point``."""

    @abstractmethod
    def get_region(
        self, center: Sequence[float], radius: float
    ) -> List[Tuple[Tuple[float, ...], Dict[str, Any]]]:
        """Return ``(point, attrs)`` tuples within ``radius`` of ``center``."""


# --------------------------------------------------------------------------- #
# EM — episodic
# --------------------------------------------------------------------------- #
class EpisodicMemoryABC(MemoryModule, ABC):
    """Time-ordered events / observation sequences.

    Implementations expose ``timescales`` (a tuple of bucket sizes in seconds)
    so callers can ask for fine vs coarse recall. WorldMM-inspired systems use
    multiple timescales (e.g. 30s / 3min / 10min / 1h); simpler
    implementations may use a single bucket ``(float("inf"),)``. Concrete
    classes should override the class attribute.
    """

    timescales: Tuple[float, ...] = (float("inf"),)

    @abstractmethod
    def append_event(self, entry: MemoryEntry) -> None:
        """Append an event, bucketing by time window per implementation."""

    @abstractmethod
    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        """Time-ordered events in ``[t_min, t_max]``."""


# --------------------------------------------------------------------------- #
# SM — semantic / knowledge
# --------------------------------------------------------------------------- #
class SemanticMemoryABC(MemoryModule, ABC):
    """Facts / rules / common sense, accessed via (subject, predicate, object)."""

    @abstractmethod
    def add_fact(self, subject: str, predicate: str, obj: Any) -> bool:
        """Add a fact triple. Returns success."""

    @abstractmethod
    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Any = None,
    ) -> List[Tuple[str, str, Any]]:
        """Return matching triples. ``None`` matches anything (don't-care)."""


# --------------------------------------------------------------------------- #
# PM — procedural / skill
# --------------------------------------------------------------------------- #
class ProceduralMemoryABC(MemoryModule, ABC):
    """Action policies / capability profile, keyed by trigger."""

    @abstractmethod
    def add_skill(self, trigger: str, skill: Any, **attrs: Any) -> bool:
        """Register a skill for ``trigger``. Returns success."""

    @abstractmethod
    def find_skill(self, trigger: str) -> Optional[Any]:
        """Return the skill stored for ``trigger`` (or ``None``)."""


__all__ = [
    "WorkingMemoryABC",
    "SceneGraphMemoryABC",
    "SpatialGeometricMemoryABC",
    "EpisodicMemoryABC",
    "SemanticMemoryABC",
    "ProceduralMemoryABC",
]
