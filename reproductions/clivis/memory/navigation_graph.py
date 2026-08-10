"""CLiViS ``NavigationGraph`` — temporal-spatial index of video segments.

Reproduces ``reproduce/CLiViS/clivis/graph/navigation_graph.py`` in pure
Python (no moviepy / cv2 / decord). The original slices video into temporal
periods; we accept period metadata from the caller.

Each period carries: description, list of areas, list of objects, optional
activity label. Areas are matched to periods via overlap between the area's
``time_range`` and the period's name (which encodes a time interval).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.query import Query, QueryResult
from unimem.core.slot_abc import SpatialGeometricMemoryABC


def _parse_range_seconds(text: str) -> Optional[Tuple[float, float]]:
    """Parse a period/range string like ``"00:00:10-00:00:30"`` to seconds."""
    if not text:
        return None
    matches = re.findall(r"(\d+):(\d+):(\d+)", text)
    if len(matches) < 2:
        return None
    def to_sec(t):
        h, m, s = t
        return int(h) * 3600 + int(m) * 60 + int(s)
    return to_sec(matches[0]), to_sec(matches[1])


def _ranges_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


class NavigationGraph(SpatialGeometricMemoryABC):
    """Period-indexed store of areas / objects / activities.

    Maps directly to the unimem Spatial slot because each period has an
    implicit spatial extent (the area it belongs to) and the queries are
    fundamentally spatial-temporal (``"what was in this area during this
    period?"``).
    """

    def __init__(
        self,
        period_description_dict: Optional[Dict[str, str]] = None,
        video_path: str = "",
        seg_output_path: str = "",
        video_duration: float = 0.0,
    ) -> None:
        self.periods_infos: Dict[str, Dict[str, Any]] = {}
        self.periods_to_obj_names: Dict[str, List[str]] = {}
        self.periods_to_activities: Dict[str, str] = {}
        self.periods_to_areas: Dict[str, List[str]] = {}
        self.person_names: Set[str] = set()
        self.area_names: Set[str] = set()
        self.obj_names: Set[str] = set()
        self.person_info: Dict[str, str] = {}
        self.video_segments_to_files: Dict[str, str] = {}
        self.video_path = video_path
        self.seg_output_path = seg_output_path
        self.video_duration = float(video_duration)

        # Seed empty periods from caller-supplied descriptions
        if period_description_dict:
            for period_name, desc in period_description_dict.items():
                self._init_period(period_name, desc)

    def _init_period(self, period_name: str, description: str) -> None:
        if period_name in self.periods_infos:
            return
        self.periods_infos[period_name] = {
            "description": description,
            "areas": [],
            "activities": [],
            "objects": [],
        }
        self.periods_to_obj_names.setdefault(period_name, [])
        self.periods_to_activities.setdefault(period_name, "")
        self.periods_to_areas.setdefault(period_name, [])

    # ------------------------------------------------------------------ #
    # Original CLiViS API
    # ------------------------------------------------------------------ #
    def get_period_names(self) -> List[str]:
        return list(self.periods_infos.keys())

    def add_persons(self, persons: List[Dict[str, str]]) -> None:
        for p in persons:
            name = p.get("name")
            if not name:
                continue
            self.person_names.add(name)
            self.person_info[name] = p.get("info", "")

    def get_person_info(self, person_name: str) -> Optional[str]:
        return self.person_info.get(person_name)

    def add_areas(self, areas: List[Dict[str, str]]) -> None:
        for a in areas:
            name = a.get("name")
            if not name:
                continue
            self.area_names.add(name)
            info = a.get("info", "")
            time_range = a.get("time_range", "")
            a_range = _parse_range_seconds(time_range)
            if a_range is None:
                continue
            # Attach to every overlapping period
            for period_name in self.periods_infos:
                p_range = _parse_range_seconds(period_name)
                if p_range is None:
                    continue
                if _ranges_overlap(a_range, p_range):
                    if name not in self.periods_to_areas[period_name]:
                        self.periods_to_areas[period_name].append(name)
                    self.periods_infos[period_name]["areas"].append(
                        {"name": name, "info": info, "time_range": time_range}
                    )

    def add_objs(self, obj_names: List[str], period: str) -> None:
        if period not in self.periods_infos:
            self._init_period(period, "")
        bucket = self.periods_to_obj_names[period]
        for o in obj_names:
            if o and o not in bucket:
                bucket.append(o)
                self.obj_names.add(o)
                self.periods_infos[period]["objects"].append(o)

    def add_activity(self, activity_name: str, period: str) -> None:
        if period not in self.periods_infos:
            self._init_period(period, "")
        self.periods_to_activities[period] = activity_name
        if activity_name and activity_name not in self.periods_infos[period]["activities"]:
            self.periods_infos[period]["activities"].append(activity_name)

    def output_periods_info(self) -> str:
        lines = []
        for period, info in self.periods_infos.items():
            lines.append(f"Period: {period}")
            lines.append(f"  description: {info['description']}")
            lines.append(f"  areas: {self.periods_to_areas.get(period, [])}")
            lines.append(f"  objects: {self.periods_to_obj_names.get(period, [])}")
            acts = self.periods_to_activities.get(period, "")
            if acts:
                lines.append(f"  activity: {acts}")
        return "\n".join(lines)

    def output_periods_description(self) -> str:
        lines = []
        for period, info in self.periods_infos.items():
            lines.append(f"Period {period}: {info['description']}")
        return "\n".join(lines)

    def get_entities_in_period(self, period: str) -> Dict[str, List[str]]:
        return {
            "persons": list(self.person_names),
            "areas": list(self.periods_to_areas.get(period, [])),
            "objects": list(self.periods_to_obj_names.get(period, [])),
        }

    def register_video_segment(self, period: str, file_path: str) -> None:
        self.video_segments_to_files[period] = file_path

    # ------------------------------------------------------------------ #
    # SpatialGeometricMemoryABC
    # ------------------------------------------------------------------ #
    def is_navigable(self, point) -> bool:
        """CLiViS doesn't compute navigability — a period either has areas or not."""
        # If ``point`` is a period string, return True iff that period has areas.
        if isinstance(point, str):
            return bool(self.periods_to_areas.get(point))
        return True

    def get_region(self, center, radius: float):
        """Return all periods whose area set includes the named ``center``.

        ``radius`` is ignored — CLiViS doesn't have metric distance; "region"
        here means "the periods in which the given area was active".
        """
        if not isinstance(center, str):
            return []
        result = []
        for period, areas in self.periods_to_areas.items():
            if center in areas:
                result.append((period, {"area": center, "areas": areas}))
        return result

    # ------------------------------------------------------------------ #
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Accept an entry describing a period/area/object observation.

        Entry schema:
            text: period description
            metadata: {
                "kind": "period" | "person" | "area" | "object" | "activity",
                "period": <period name>,
                "name": <entity name>,
                "info": <optional info>,
                "time_range": <optional, for area>,
            }
        """
        kind = entry.metadata.get("kind", "period")
        period = entry.metadata.get("period")
        name = entry.metadata.get("name", "")
        if kind == "period":
            self._init_period(period or entry.entry_id, entry.text)
        elif kind == "person":
            self.add_persons([{"name": name, "info": entry.metadata.get("info", "")}])
        elif kind == "area":
            self.add_areas([{
                "name": name,
                "info": entry.metadata.get("info", ""),
                "time_range": entry.metadata.get("time_range", ""),
            }])
        elif kind == "object":
            if period:
                self.add_objs([name], period)
        elif kind == "activity":
            if period:
                self.add_activity(name, period)
        else:
            return False
        return True

    def read(self, query: Query) -> QueryResult:
        """Retrieve period entries filtered by area/object via semantic_keys."""
        sem = set(query.semantic or [])
        entries = []
        for period, info in self.periods_infos.items():
            period_keys = set(self.periods_to_areas.get(period, [])) | set(
                self.periods_to_obj_names.get(period, [])
            )
            if sem and not sem.issubset(period_keys):
                continue
            entries.append(MemoryEntry(
                entry_id=f"nav-{period}",
                text=info["description"],
                semantic_keys=sorted(period_keys),
                temporal_keys=[(_parse_range_seconds(period) or (0.0, 0.0))[0]],
                source_slot="spatial_geometric",
                metadata={"period": period, **self.get_entities_in_period(period)},
            ))
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot="spatial_geometric")

    def clear(self) -> None:
        self.periods_infos.clear()
        self.periods_to_obj_names.clear()
        self.periods_to_activities.clear()
        self.periods_to_areas.clear()
        self.person_names.clear()
        self.area_names.clear()
        self.obj_names.clear()
        self.person_info.clear()
        self.video_segments_to_files.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.periods_infos),
            "n_persons": len(self.person_names),
            "n_areas": len(self.area_names),
            "n_objects": len(self.obj_names),
        }


__all__ = ["NavigationGraph", "_parse_range_seconds"]
