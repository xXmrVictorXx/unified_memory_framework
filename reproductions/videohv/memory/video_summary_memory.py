"""``VideoSummaryMemory`` — storage-backed per-clip episodic memory.

A thin facade over :class:`~unimem.graph_storage.base.GraphStorage` that
preserves VideoHV-Agent's high-level API (``ingest``, ``get_clip``,
``get_timeline``, ``get_object_tags``) while delegating storage to a
portable backend.

Differences from the legacy implementation:

* **Inherits from :class:`~unimem.core.module.MemoryModule`** (not the
  ``EpisodicMemoryABC`` mixin) so the
  :class:`~unimem.graph.graph.MemoryGraph` can host it as a node.
* **Clips live in :class:`GraphStorage`** as nodes labelled
  ``:episodic:VideoClip``.
* **Each clip carries a ``:TimeIndex`` node** attached via ``:AT_TIME``
  with ``clip_index``, ``start_t``, ``end_t`` — enabling graph-native
  time queries via Cypher.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry, MultiAxisIndex
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage
from unimem.graph_storage.time_index import attach_clip

VIDEO_CLIP_LABEL = "VideoClip"


class VideoSummaryMemory(MemoryModule):
    """Episodic store of per-clip summaries.

    Construction
    ------------
    Accepts the three pre-computed lists straight from VideoHV-Agent's
    bundle: ``action_captions``, ``object_detections``, ``clip_boundaries``.
    If ``clip_boundaries`` is shorter than the caption list, defaults to
    integer clip-index timestamps ``(i, i+1)``.

    Each clip becomes one :class:`MemoryEntry` persisted to the bound
    :class:`GraphStorage` with:

    * node labels: ``["episodic", "VideoClip"]``
    * node properties: ``text``, ``semantic_keys``, ``temporal_keys``,
      ``clip_index``, ``start_t``, ``end_t``
    * an attached ``:TimeIndex`` node connected via ``:AT_TIME`` carrying
      ``clip_index`` / ``start_t`` / ``end_t`` for graph-native time queries
    """

    SLOT = MemorySlot.EM

    def __init__(
        self,
        action_captions: Optional[Sequence[str]] = None,
        object_detections: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
        clip_boundaries: Optional[Sequence[Tuple[float, float]]] = None,
        graph_storage: Optional[GraphStorage] = None,
    ) -> None:
        super().__init__(slot=self.SLOT)
        self._gs = graph_storage or InMemoryGraphStorage()
        # Legacy in-memory mirror kept for fast / API-compatible access.
        self._entries: List[MemoryEntry] = []
        self._index = MultiAxisIndex()
        self._by_id: Dict[str, MemoryEntry] = {}

        if action_captions:
            self.ingest(action_captions, object_detections, clip_boundaries)

    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        action_captions: Sequence[str],
        object_detections: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
        clip_boundaries: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> int:
        """Bulk-add clips. Returns the number of clips ingested."""
        n = len(action_captions)
        object_detections = list(object_detections or [])
        clip_boundaries = list(clip_boundaries or [])
        for i in range(n):
            caption = action_captions[i]
            objs = object_detections[i] if i < len(object_detections) else []
            boundaries = (
                clip_boundaries[i] if i < len(clip_boundaries) else (float(i), float(i + 1))
            )
            try:
                start_t, end_t = float(boundaries[0]), float(boundaries[1])
            except (TypeError, IndexError, ValueError):
                start_t, end_t = float(i), float(i + 1)
            labels = self._extract_labels(objs)
            entry = MemoryEntry(
                entry_id=f"clip-{i}",
                text=caption,
                semantic_keys=[f"clip-{i}", *labels],
                temporal_keys=[start_t, end_t],
                source_slot=self.SLOT.value,
                metadata={
                    "clip_index": i,
                    "object_detections": list(objs),
                    "start_t": start_t,
                    "end_t": end_t,
                },
            )
            self._add_entry(entry)
        return n

    def _add_entry(self, entry: MemoryEntry) -> None:
        if entry.entry_id in self._by_id:
            self._index.remove(entry.entry_id)
            self._entries = [e for e in self._entries if e.entry_id != entry.entry_id]
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        self._index.add(entry)
        # Persist to storage with clip sub-label + TimeIndex attachment.
        clip_idx = entry.metadata.get("clip_index")
        start_t = entry.metadata.get("start_t")
        end_t = entry.metadata.get("end_t")
        self._gs.add_node(
            entry.entry_id,
            [self.SLOT.value, VIDEO_CLIP_LABEL],
            {
                "text": entry.text,
                "semantic_keys": list(entry.semantic_keys),
                "temporal_keys": list(entry.temporal_keys),
                "metadata": dict(entry.metadata),
                "source_slot": entry.source_slot,
                "clip_index": clip_idx,
                "start_t": start_t,
                "end_t": end_t,
            },
        )
        if clip_idx is not None:
            attach_clip(
                self._gs,
                entry.entry_id,
                clip_idx,
                start_t=start_t,
                end_t=end_t,
            )

    @staticmethod
    def _extract_labels(objs: Sequence[Dict[str, Any]]) -> List[str]:
        labels: List[str] = []
        for o in objs:
            label = o.get("label") or o.get("name") or o.get("class")
            if label and label not in labels:
                labels.append(str(label).lower())
        return labels

    # ------------------------------------------------------------------ #
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        self._add_entry(entry)
        return True

    def read(self, query: Query) -> QueryResult:
        if query.semantic:
            ids = self._index.lookup(
                semantic=query.semantic,
                t_min=query.t_min,
                t_max=query.t_max,
                require_any_axis=True,
            )
        elif query.has_temporal:
            ids = self._index.lookup_temporal(query.t_min, query.t_max)
        else:
            ids = set(self._by_id.keys())
        entries = [self._by_id[i] for i in ids if i in self._by_id]
        entries.sort(key=lambda e: e.metadata.get("clip_index", 0))
        if query.top_k is not None:
            entries = entries[: query.top_k]
        return QueryResult(entries=entries, source_slot=self.SLOT.value)

    def clear(self) -> None:
        self._entries.clear()
        self._by_id.clear()
        self._index.clear()
        # Drop every VideoClip + TimeIndex node attached to this slot.
        self._gs.query(
            f"MATCH (n:{self.SLOT.value}:{VIDEO_CLIP_LABEL}) DETACH DELETE n"
        )

    def stats(self) -> Dict[str, Any]:
        tag_counts: Dict[str, int] = {}
        for e in self._entries:
            for k in e.semantic_keys:
                if k.startswith("clip-"):
                    continue
                tag_counts[k] = tag_counts.get(k, 0) + 1
        return {
            "count": len(self._entries),
            "n_distinct_object_tags": len(tag_counts),
            "top_tags": sorted(tag_counts.items(), key=lambda x: -x[1])[:5],
        }

    # ------------------------------------------------------------------ #
    # Episodic-style helpers (preserved from the legacy API)
    # ------------------------------------------------------------------ #
    def append_event(self, entry: MemoryEntry) -> None:
        self._add_entry(entry)

    def get_timeline(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None
    ) -> List[MemoryEntry]:
        if t_min is None and t_max is None:
            return list(self._entries)
        ids = self._index.lookup_temporal(t_min, t_max)
        out = [self._by_id[i] for i in ids if i in self._by_id]
        out.sort(key=lambda e: min(e.temporal_keys) if e.temporal_keys else 0.0)
        return out

    def get_clip(self, clip_index: int) -> Optional[MemoryEntry]:
        return self._by_id.get(f"clip-{clip_index}")

    def get_object_tags(self) -> List[str]:
        tags: set = set()
        for e in self._entries:
            for k in e.semantic_keys:
                if not k.startswith("clip-"):
                    tags.add(k)
        return sorted(tags)


__all__ = ["VideoSummaryMemory", "VIDEO_CLIP_LABEL"]
