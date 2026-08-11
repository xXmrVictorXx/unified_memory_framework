"""CLiViS ``NavigationGraph`` — storage-backed temporal-spatial index.

A thin facade over :class:`~unimem.graph_storage.base.GraphStorage` that
preserves CLiViS's high-level API (period indexing, area/object/activity
lookup) without subclassing any unimem slot ABC. It still inherits from
:class:`~unimem.core.module.MemoryModule` so the
:class:`~unimem.graph.graph.MemoryGraph` can host it as a node.

Each period becomes a node labelled ``:spatial_geometric:Period``; areas /
objects / persons / activities are tracked via per-period edge relationships
(``IN_PERIOD`` / ``HAS_AREA`` / ``HAS_OBJECT``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage


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


class NavigationGraph(MemoryModule):
    """Period-indexed store of areas / objects / activities.

    Each period is stored as a graph node ``:spatial_geometric:Period``
    keyed by its name. Areas, persons, objects, activities are tracked
    as edges from the period node (``HAS_AREA``, ``HAS_OBJECT``, etc.).
    """

    SLOT = MemorySlot.GM  # Spatial-geometric slot

    def __init__(
        self,
        period_description_dict: Optional[Dict[str, str]] = None,
        video_path: str = "",
        seg_output_path: str = "",
        video_duration: float = 0.0,
        graph_storage: Optional[GraphStorage] = None,
    ) -> None:
        super().__init__(slot=self.SLOT)
        self._gs = graph_storage or InMemoryGraphStorage()
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

        if period_description_dict:
            for period_name, desc in period_description_dict.items():
                self._init_period(period_name, desc)

    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

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
        # Mirror to storage.
        self._gs.add_node(
            period_name,
            [self.SLOT.value, "Period"],
            {"description": description},
        )

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
            self._gs.add_node(
                name,
                [self.SLOT.value, "Person"],
                {"info": p.get("info", "")},
            )

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
            self._gs.add_node(
                name,
                [self.SLOT.value, "Area"],
                {"info": info, "time_range": time_range},
            )
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
                self._gs.add_node(
                    o,
                    [self.SLOT.value, "Object"],
                    {"period": period},
                )
                self._gs.add_edge(period, o, "HAS_OBJECT", {})

    def add_activity(self, activity_name: str, period: str) -> None:
        if period not in self.periods_infos:
            self._init_period(period, "")
        self.periods_to_activities[period] = activity_name
        if activity_name and activity_name not in self.periods_infos[period]["activities"]:
            self.periods_infos[period]["activities"].append(activity_name)

    def output_periods_info(self) -> str:
        lines: List[str] = []
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
        lines: List[str] = []
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
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self.periods_infos),
            "n_persons": len(self.person_names),
            "n_areas": len(self.area_names),
            "n_objects": len(self.obj_names),
        }

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
        # Drop every Period / Area / Object / Person node labelled with slot.
        self._gs.query(
            f"MATCH (n:{self.SLOT.value}) DETACH DELETE n"
        )

    # ------------------------------------------------------------------ #
    # MemoryModule contract — dispatches on entry.metadata["kind"]
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
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
        sem = set(query.semantic or [])
        entries: List[MemoryEntry] = []
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
                source_slot=self.SLOT.value,
                metadata={"period": period, **self.get_entities_in_period(period)},
            ))
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot=self.SLOT.value)


__all__ = ["NavigationGraph", "_parse_range_seconds"]
