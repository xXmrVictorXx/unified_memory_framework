"""Minimal SLAM map interface used by R4.

The paper uses MapAnything (Keetha et al. 2025) as its SLAM backend. We do
*not* reproduce MapAnything here; instead we model the *interface* R4 needs
from M:

* a global Euclidean frame
* an ego-pose history (per-timestep agent position)
* "special points" — object centroids inserted into M as pointers to
  ``ObjectRecord`` ids

This is enough to support R4's spatial retrieval (nearest-neighbor and
directional filtering). A real backend can be swapped in by subclassing or
by passing a compatible object via ``R4KnowledgeDatabase(slam_map=...)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class SimpleSLAMMap:
    """In-memory stub of MapAnything's interface for R4's purposes.

    The map is just a global Euclidean frame with a set of named "special
    points" (object centroids) and an ego-pose trajectory. No occupancy grid
    or pose-graph optimisation — those are irrelevant to R4's retrieval
    semantics and would require the real MapAnything.
    """

    def __init__(self) -> None:
        # object_id -> 3D centroid (the "special point")
        self._special_points: Dict[str, Tuple[float, float, float]] = {}
        # ego pose trajectory: list of (t, x, y, z)
        self._trajectory: List[Tuple[float, float, float, float]] = []

    def add_special_point(self, object_id: str, centroid: Sequence[float]) -> None:
        self._special_points[object_id] = tuple(float(c) for c in centroid)

    def remove_special_point(self, object_id: str) -> None:
        self._special_points.pop(object_id, None)

    def update_centroid(self, object_id: str, centroid: Sequence[float]) -> None:
        self._special_points[object_id] = tuple(float(c) for c in centroid)

    def get_centroid(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        return self._special_points.get(object_id)

    def all_special_points(self) -> Dict[str, Tuple[float, float, float]]:
        return dict(self._special_points)

    def append_ego_pose(
        self, t: float, position: Sequence[float]
    ) -> None:
        self._trajectory.append(
            (float(t), float(position[0]), float(position[1]), float(position[2]))
        )

    def nearest_neighbors(
        self,
        query: Sequence[float],
        k: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Return ``[(object_id, distance), ...]`` sorted ascending.

        Filters by ``radius`` (Euclidean) and truncates to ``k`` if given.
        """
        q = tuple(float(c) for c in query)
        scored: List[Tuple[str, float]] = []
        for oid, c in self._special_points.items():
            d = sum((a - b) ** 2 for a, b in zip(c, q)) ** 0.5
            if radius is not None and d > radius:
                continue
            scored.append((oid, d))
        scored.sort(key=lambda x: x[1])
        if k is not None:
            scored = scored[:k]
        return scored

    def directional_filter(
        self,
        ego: Sequence[float],
        direction: Sequence[float],
        half_angle_deg: float = 60.0,
    ) -> List[str]:
        """Return object ids whose centroid lies within ``half_angle_deg`` of
        ``direction`` as seen from ``ego`` (both in world coords)."""
        import math

        e = tuple(float(c) for c in ego)
        d = tuple(float(c) for c in direction)
        d_norm = sum(x * x for x in d) ** 0.5 or 1.0
        d = tuple(x / d_norm for x in d)
        cos_threshold = math.cos(math.radians(half_angle_deg))
        kept: List[str] = []
        for oid, c in self._special_points.items():
            v = tuple(c[i] - e[i] for i in range(3))
            v_norm = sum(x * x for x in v) ** 0.5
            if v_norm == 0:
                continue
            cos_sim = sum(v[i] * d[i] for i in range(3)) / v_norm
            if cos_sim >= cos_threshold:
                kept.append(oid)
        return kept


__all__ = ["SimpleSLAMMap"]
