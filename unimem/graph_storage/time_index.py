"""Time-index node helpers.

The TimeIndex pattern encodes temporal anchoring as a separate ``:TimeIndex``
node connected to memory nodes via ``:AT_TIME`` relationships. This keeps
time-based queries index-friendly in graph databases (Neo4j can index node
properties + relationship type) instead of relying on full scans over memory
nodes.

These helpers provide convenient builders for the canonical time-index
shapes used across CLiViS (period), VideoHV (clip_index), and
ListEpisodicMemory (timescale bucket).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..core.entry import MemoryEntry
from ..core.slots import MemorySlot
from .base import GraphStorage


def attach_timestamp(
    storage: GraphStorage,
    memory_node_id: str,
    timestamp: float,
    *,
    extra_props: Optional[Dict[str, Any]] = None,
) -> str:
    """Attach a single-timestamp TimeIndex node.

    Returns the TimeIndex node id.
    """
    props: Dict[str, Any] = {"timestamp": float(timestamp)}
    if extra_props:
        props.update(extra_props)
    time_id = storage._time_index_id(memory_node_id, props)  # noqa: SLF001
    storage.add_node(time_id, ["TimeIndex"], props)
    storage.add_edge(memory_node_id, time_id, "AT_TIME", {})
    return time_id


def attach_clip(
    storage: GraphStorage,
    memory_node_id: str,
    clip_index: int,
    *,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
    period: Optional[str] = None,
) -> str:
    """Attach a clip-oriented TimeIndex node (VideoHV style)."""
    props: Dict[str, Any] = {"clip_index": int(clip_index)}
    if start_t is not None:
        props["start_t"] = float(start_t)
    if end_t is not None:
        props["end_t"] = float(end_t)
    if period is not None:
        props["period"] = str(period)
    time_id = storage._time_index_id(memory_node_id, props)  # noqa: SLF001
    storage.add_node(time_id, ["TimeIndex"], props)
    storage.add_edge(memory_node_id, time_id, "AT_TIME", {})
    return time_id


def attach_period(
    storage: GraphStorage,
    memory_node_id: str,
    period: str,
    *,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
) -> str:
    """Attach a period-oriented TimeIndex node (CLiViS style)."""
    props: Dict[str, Any] = {"period": str(period)}
    if start_t is not None:
        props["start_t"] = float(start_t)
    if end_t is not None:
        props["end_t"] = float(end_t)
    time_id = storage._time_index_id(memory_node_id, props)  # noqa: SLF001
    storage.add_node(time_id, ["TimeIndex"], props)
    storage.add_edge(memory_node_id, time_id, "AT_TIME", {})
    return time_id


def attach_timescale_bucket(
    storage: GraphStorage,
    memory_node_id: str,
    timescale: float,
    *,
    bucket_start: Optional[float] = None,
    bucket_end: Optional[float] = None,
) -> str:
    """Attach a timescale-bucket TimeIndex node (WorldMM / ListEpisodicMemory)."""
    props: Dict[str, Any] = {"timescale": float(timescale), "kind": "bucket"}
    if bucket_start is not None:
        props["start_t"] = float(bucket_start)
    if bucket_end is not None:
        props["end_t"] = float(bucket_end)
    time_id = storage._time_index_id(memory_node_id, props)  # noqa: SLF001
    storage.add_node(time_id, ["TimeIndex"], props)
    storage.add_edge(memory_node_id, time_id, "AT_TIME", {})
    return time_id


def write_with_time_index(
    storage: GraphStorage,
    slot: MemorySlot,
    entry: MemoryEntry,
    *,
    clip_index: Optional[int] = None,
    timestamp: Optional[float] = None,
    period: Optional[str] = None,
    start_t: Optional[float] = None,
    end_t: Optional[float] = None,
    extra_labels: Optional[list] = None,
) -> str:
    """Write a memory node + attach the requested TimeIndex shape.

    Exactly one of {clip_index, timestamp, period} should be provided (the
    function picks the most specific one). ``start_t`` / ``end_t`` can be
    added to any of the shapes for range queries.
    """
    storage.add_memory_node(slot, entry, extra_labels=extra_labels)
    if clip_index is not None:
        attach_clip(
            storage,
            entry.entry_id,
            clip_index,
            start_t=start_t,
            end_t=end_t,
            period=period,
        )
    elif period is not None:
        attach_period(
            storage, entry.entry_id, period, start_t=start_t, end_t=end_t
        )
    elif timestamp is not None:
        attach_timestamp(
            storage,
            entry.entry_id,
            timestamp,
            extra_props={"start_t": start_t, "end_t": end_t}
            if (start_t is not None or end_t is not None)
            else None,
        )
    return entry.entry_id


__all__ = [
    "attach_timestamp",
    "attach_clip",
    "attach_period",
    "attach_timescale_bucket",
    "write_with_time_index",
]
