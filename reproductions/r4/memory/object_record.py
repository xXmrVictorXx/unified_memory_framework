"""R4 object record — the atomic unit of the 4D knowledge database.

Paper (Eq. 3): ``O_j^t = { SEM, SPA, TEM }``.

We keep these as plain dataclasses (not unimem `MemoryEntry` subclasses) so
the *storage layer* can be reasoned about independently from the
*framework-facing* wrapper (see ``knowledge_db.py`` for the MemoryModule
adapter that translates between the two).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class SemanticAxis:
    """The SEM axis: natural-language description + its embedding vector."""

    description: str
    embedding: Optional[Sequence[float]] = None  # lazily populated by DB

    def __post_init__(self) -> None:
        if self.embedding is not None:
            self.embedding = list(self.embedding)


@dataclass
class SpatialAxis:
    """The SPA axis: 3D world-coordinate centroid + bounding-box extent."""

    centroid: Tuple[float, float, float]
    extent: Tuple[float, float, float]

    def __post_init__(self) -> None:
        self.centroid = tuple(float(c) for c in self.centroid)
        self.extent = tuple(float(e) for e in self.extent)
        if len(self.centroid) != 3 or len(self.extent) != 3:
            raise ValueError("centroid and extent must be 3-tuples")


@dataclass
class TemporalAxis:
    """The TEM axis: observation timestamps; first=appearance, last=most-recent."""

    timestamps: List[float] = field(default_factory=list)

    def observe(self, t: float) -> None:
        t = float(t)
        # Keep sorted so first/last are meaningful; tolerate out-of-order arrival.
        if t not in self.timestamps:
            self.timestamps.append(t)
            self.timestamps.sort()

    @property
    def first_seen(self) -> Optional[float]:
        return self.timestamps[0] if self.timestamps else None

    @property
    def last_seen(self) -> Optional[float]:
        return self.timestamps[-1] if self.timestamps else None

    def overlaps(self, t_min: Optional[float], t_max: Optional[float]) -> bool:
        if not self.timestamps:
            return False
        if t_min is None and t_max is None:
            return True
        lo = float(t_min) if t_min is not None else float("-inf")
        hi = float(t_max) if t_max is not None else float("inf")
        return any(lo <= t <= hi for t in self.timestamps)


@dataclass
class ObjectRecord:
    """A single object entry ``O_j`` in the knowledge database."""

    unique_id: str
    sem: SemanticAxis
    spa: SpatialAxis
    tem: TemporalAxis = field(default_factory=TemporalAxis)
    # Optional provenance / extra attributes; not part of the paper's formal
    # spec but useful for downstream debugging (e.g. source mask hash, sensor).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Compact text rendering used when serialising retrieval context."""
        ts = self.tem.timestamps
        ts_str = ", ".join(f"{t:.1f}" for t in ts[:3])
        if len(ts) > 3:
            ts_str += f", ... (+{len(ts)-3} more)"
        return (
            f"object {self.unique_id}: {self.sem.description} "
            f"@ world_pos=({self.spa.centroid[0]:.2f},{self.spa.centroid[1]:.2f},"
            f"{self.spa.centroid[2]:.2f}) "
            f"extent=({self.spa.extent[0]:.2f},{self.spa.extent[1]:.2f},"
            f"{self.spa.extent[2]:.2f}) "
            f"observed_at=[{ts_str}]"
        )


__all__ = ["SemanticAxis", "SpatialAxis", "TemporalAxis", "ObjectRecord"]
