"""R4 knowledge database — facade over :class:`MemoryModule` + :class:`DedupPolicy`.

R4's ``D = {M, {O_j}}`` maps onto unimem's :class:`MemoryModule` (slot=GM)
plus a :class:`DedupPolicy` (Eq. 5 deduplication). This module provides the
R4-specific API (``observe_object`` / ``get_record`` / ``_retrieve``) on top
of those generic components.

Compared to the previous implementation, no subclassing is required —
``R4KnowledgeDatabase`` is a thin wrapper that holds:

* a :class:`GraphStorage` (for object node storage + spatial property lookup)
* a :class:`VectorStorage` (for SEM embedding ANN search)
* a :class:`MemoryModule` (for unimem integration, wired with the DedupPolicy)
* a :class:`SimpleSLAMMap` (for spatial nearest-neighbour index)

This means the storage layout is portable: the same GraphStorage +
VectorStorage can be reused by any other module that wants to query R4's
objects (e.g. a scene-graph module that adds labels to the same nodes).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.core.slot_abc import SpatialGeometricMemoryABC
from unimem.graph_storage import GraphStorage, InMemoryGraphStorage
from unimem.policies.write_policy import DedupPolicy
from unimem.vector_storage import InMemoryVectorStorage, VectorStorage

from .embedding import EmbeddingFn, cosine_similarity, get_default_embedding
from .object_record import ObjectRecord, SemanticAxis, SpatialAxis, TemporalAxis
from .slam_map import SimpleSLAMMap


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bbox_to_centroid_extent(
    mask_points: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute (centroid, extent) from a set of 3D points.

    Mirrors Eqs. (1) and (2) of the paper: centroid = mean, extent = per-axis
    max-min (i.e. axis-aligned bounding box dimensions).
    """
    if not mask_points:
        raise ValueError("mask_points must be non-empty")
    pts = [tuple(float(c) for c in p) for p in mask_points]
    if any(len(p) != 3 for p in pts):
        raise ValueError("all mask_points must be 3D")
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    ex = max(p[0] for p in pts) - min(p[0] for p in pts)
    ey = max(p[1] for p in pts) - min(p[1] for p in pts)
    ez = max(p[2] for p in pts) - min(p[2] for p in pts)
    return (cx, cy, cz), (ex, ey, ez)


R4_COLLECTION = "r4_objects"
R4_OBJECT_LABEL = "r4_object"


class R4KnowledgeDatabase(SpatialGeometricMemoryABC):
    """R4's 4D knowledge database as a thin facade.

    Construction
    ------------
    graph_storage / vector_storage:
        Optional pre-built backends. If omitted, fresh in-memory ones are
        created.
    embedding_fn:
        Callable mapping text → vector. Defaults to
        :func:`~reproductions.r4.memory.embedding.get_default_embedding`.
    eps_c / delta_s:
        Eq. 5 dedup thresholds.
    slam_map:
        Optional pre-built SLAM map. If omitted, a fresh
        :class:`SimpleSLAMMap` is used.
    """

    DEFAULT_EPS_C = 0.5  # metres
    DEFAULT_DELTA_S = 0.7  # cosine similarity
    COLLECTION = R4_COLLECTION

    def __init__(
        self,
        embedding_fn: Optional[EmbeddingFn] = None,
        graph_storage: Optional[GraphStorage] = None,
        vector_storage: Optional[VectorStorage] = None,
        slam_map: Optional[SimpleSLAMMap] = None,
        eps_c: float = DEFAULT_EPS_C,
        delta_s: float = DEFAULT_DELTA_S,
        vlm_describe: Optional[Any] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        # Initialise the underlying MemoryModule bookkeeping (slot, policies).
        super().__init__(slot=MemorySlot.GM)
        # Store config + backends
        self._embedding_fn = embedding_fn or get_default_embedding()
        self.slam_map = slam_map or SimpleSLAMMap()
        self.eps_c = float(eps_c)
        self.delta_s = float(delta_s)
        self._vlm_describe = vlm_describe
        self._vector_dim = vector_dim

        self._gs = graph_storage or InMemoryGraphStorage()
        self._vs = vector_storage or InMemoryVectorStorage()
        # Lazy collection creation (deferred until we know the embedding dim).
        self._collection_ready = False

        # Build the unimem module facade (we delegate write/read to it).
        self._module = MemoryModule(
            slot=MemorySlot.GM,
            graph_storage=self._gs,
            write_policy=DedupPolicy(
                vector_storage=self._vs,
                graph_storage=self._gs,
                collection=self.COLLECTION,
                embedding_fn=self._embedding_fn,
                eps_c=self.eps_c,
                delta_s=self.delta_s,
                vector_dim=vector_dim,
            ),
        )
        self._next_id = 1
        self._last_query_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def graph_storage(self) -> GraphStorage:
        return self._gs

    @property
    def vector_storage(self) -> VectorStorage:
        return self._vs

    @property
    def module(self) -> MemoryModule:
        return self._module

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_ready:
            return
        actual_dim = self._vector_dim or dim
        if not self._vs.collection_exists(self.COLLECTION):
            self._vs.create_collection(
                self.COLLECTION,
                vector_dim=actual_dim,
                distance_metric="cosine",
            )
        self._collection_ready = True

    # ------------------------------------------------------------------ #
    # SpatialGeometricMemoryABC required methods
    # ------------------------------------------------------------------ #
    def is_navigable(self, point: Sequence[float]) -> bool:
        return True  # stub: R4 delegates this to MapAnything

    def get_region(
        self, center: Sequence[float], radius: float
    ) -> List[Tuple[Tuple[float, ...], Dict[str, Any]]]:
        out: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []
        for oid, d in self.slam_map.nearest_neighbors(center, radius=float(radius)):
            rec = self.get_record(oid)
            if rec is None:
                continue
            out.append((rec.spa.centroid, {
                "object_id": oid,
                "description": rec.sem.description,
                "distance": d,
            }))
        return out

    # ------------------------------------------------------------------ #
    # MemoryModule contract — delegate to the inner module.
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        # R4 requires a spatial anchor (centroid). Reject entries without one.
        if not entry.spatial_keys:
            return False
        # Pre-flight: stamp R4 metadata so DedupPolicy can find the centroid.
        entry.metadata.setdefault("_r4_label", R4_OBJECT_LABEL)
        # The underlying module + DedupPolicy decide whether to admit.
        ok = self._module.write(entry, context)
        if ok:
            # Add the R4-specific sub-label and ensure the SLAM map has the
            # centroid registered as a "special point".
            spa = entry.spatial_keys[0]
            centroid = spa[:3] if len(spa) >= 3 else None
            extent = spa[3:6] if len(spa) >= 6 else [0.1, 0.1, 0.1]
            self._gs.add_node(
                entry.entry_id,
                [MemorySlot.GM.value, R4_OBJECT_LABEL],
                {
                    "description": entry.metadata.get("description", entry.text),
                    "centroid": list(centroid) if centroid else [],
                    "extent": list(extent),
                    "first_seen": entry.temporal_keys[0]
                    if entry.temporal_keys
                    else None,
                    "last_seen": entry.temporal_keys[-1]
                    if entry.temporal_keys
                    else None,
                },
            )
            if centroid:
                self.slam_map.add_special_point(entry.entry_id, centroid)
        return ok

    def read(self, query: Query) -> QueryResult:
        # Three-axis retrieval: SEM (semantic) ∩ SPA (spatial) ∩ TEM.
        ids = self._retrieve(
            k_sem=query.semantic or None,
            k_spa_centroid=(query.spatial[0] if query.spatial else None),
            k_spa_radius=query.metadata.get("spatial_radius"),
            k_spa_k=query.metadata.get("spatial_k"),
            k_t_min=query.t_min,
            k_t_max=query.t_max,
        )
        recs = [r for r in (self.get_record(i) for i in ids) if r is not None]
        entries = [self._record_to_entry(r) for r in recs]
        if query.top_k is not None:
            entries = entries[: query.top_k]
        self._last_query_summary = {
            "n_matches": len(entries),
            "k_sem": query.semantic,
            "k_spa_centroid": query.spatial[0] if query.spatial else None,
            "t_range": (query.t_min, query.t_max),
        }
        return QueryResult(entries=entries, source_slot=MemorySlot.GM.value)

    def clear(self) -> None:
        # Drop all R4 object nodes from the graph storage.
        self._gs.query(
            f"MATCH (n:{MemorySlot.GM.value}:{R4_OBJECT_LABEL}) DETACH DELETE n"
        )
        # Clear the vector collection
        if self._vs.collection_exists(self.COLLECTION):
            self._vs.delete_collection(self.COLLECTION)
        self._collection_ready = False
        self.slam_map = SimpleSLAMMap()
        self._next_id = 1

    def stats(self) -> Dict[str, Any]:
        # Count R4 object nodes
        rows = self._gs.query(
            f"MATCH (n:{R4_OBJECT_LABEL}) RETURN COUNT(*)"
        )
        count = rows[0].get("count", 0) if rows else 0
        return {
            "count": int(count),
            "n_slam_points": len(self.slam_map.all_special_points()),
            "eps_c": self.eps_c,
            "delta_s": self.delta_s,
            "last_query": self._last_query_summary,
        }

    # ------------------------------------------------------------------ #
    # R4-specific API
    # ------------------------------------------------------------------ #
    def observe_object(
        self,
        description: str,
        centroid: Sequence[float],
        extent: Sequence[float],
        timestamp: float,
        mask_points: Optional[Sequence[Sequence[float]]] = None,
        object_id: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Add or merge an object observation via the storage-backed module."""
        # Pick / mint an object id
        if object_id is None:
            object_id = self._mint_id()
        entry = MemoryEntry(
            entry_id=object_id,
            text=description,
            spatial_keys=[(*centroid, *extent)],
            temporal_keys=[float(timestamp)],
            source_slot=MemorySlot.GM.value,
            metadata={
                "description": description,
                "centroid": list(centroid),
                "extent": list(extent),
                "first_seen": float(timestamp),
                "last_seen": float(timestamp),
                "confidence": confidence,
                "_r4_label": R4_OBJECT_LABEL,
            },
        )
        ok = self.write(entry, MemoryContext(timestamp=timestamp))
        # ``write`` returns False on dedup-merge — still considered a
        # successful observation, just not a *new* record.
        return ok

    def _mint_id(self) -> str:
        # Avoid colliding with existing records.
        while True:
            candidate = f"obj-{self._next_id:04d}"
            self._next_id += 1
            if self._gs.get_node(candidate) is None:
                return candidate

    def get_record(self, object_id: str) -> Optional[ObjectRecord]:
        node = self._gs.get_node(object_id)
        if node is None or R4_OBJECT_LABEL not in node["labels"]:
            return None
        props = node["properties"]
        # Prefer explicit r4 metadata (centroid / extent / description) which
        # ``observe_object`` populates; fall back to the multi-axis fields.
        try:
            c = props.get("centroid") or (
                props.get("spatial_keys", [[0, 0, 0]])[0][:3]
                if props.get("spatial_keys")
                else [0, 0, 0]
            )
            e = props.get("extent") or (
                list(props.get("spatial_keys", [[0, 0, 0]])[0][3:6])
                if props.get("spatial_keys")
                and len(props.get("spatial_keys", [[]])[0]) >= 6
                else [0.1, 0.1, 0.1]
            )
            centroid = tuple(float(x) for x in c)
            extent = tuple(float(x) for x in e)
        except (TypeError, ValueError, IndexError):
            return None
        # temporal_keys is updated by DedupPolicy merge; honour it.
        tk = props.get("temporal_keys") or []
        timestamps = sorted(float(t) for t in tk)
        if not timestamps and props.get("first_seen") is not None:
            timestamps = [float(props["first_seen"])]
        description = (
            props.get("description") or props.get("text") or ""
        )
        return ObjectRecord(
            unique_id=object_id,
            sem=SemanticAxis(description=description, embedding=None),
            spa=SpatialAxis(centroid=centroid, extent=extent),
            tem=TemporalAxis(timestamps=timestamps),
            metadata={
                k: v
                for k, v in props.items()
                if k
                not in (
                    "description",
                    "centroid",
                    "extent",
                    "first_seen",
                    "last_seen",
                    "_r4_label",
                    "entry_id",
                    "text",
                    "semantic_keys",
                    "spatial_keys",
                    "temporal_keys",
                    "metadata",
                    "source_slot",
                )
            },
        )

    def all_records(self) -> List[ObjectRecord]:
        rows = self._gs.query(
            f"MATCH (n:{R4_OBJECT_LABEL}) RETURN n.node_id AS id"
        )
        recs = []
        for r in rows:
            rec = self.get_record(r["id"])
            if rec is not None:
                recs.append(rec)
        return recs

    # ------------------------------------------------------------------ #
    # Retrieval internals — R4 §3.2
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        k_sem: Optional[Sequence[str]] = None,
        k_spa_centroid: Optional[Sequence[float]] = None,
        k_spa_radius: Optional[float] = None,
        k_spa_k: Optional[int] = None,
        k_t_min: Optional[float] = None,
        k_t_max: Optional[float] = None,
    ) -> List[str]:
        # SEM axis: union over multiple semantic keys, each via cosine sim.
        if k_sem:
            self._ensure_collection(len(self._embedding_fn(k_sem[0])))
            sem_sets = [self._semantic_search(s) for s in k_sem]
            sem_ids: Optional[set] = set()
            for s in sem_sets:
                sem_ids |= s
        else:
            sem_ids = None

        # SPA axis: nearest neighbors
        if k_spa_centroid is not None:
            spa_ids = {
                oid
                for oid, _ in self.slam_map.nearest_neighbors(
                    k_spa_centroid, k=k_spa_k, radius=k_spa_radius
                )
            }
        else:
            spa_ids = None

        # TEM axis: temporal overlap
        if k_t_min is not None or k_t_max is not None:
            tem_ids = set()
            for rec in self.all_records():
                if rec.tem.overlaps(k_t_min, k_t_max):
                    tem_ids.add(rec.unique_id)
        else:
            tem_ids = None

        per_axis = [s for s in (sem_ids, spa_ids, tem_ids) if s is not None]
        if not per_axis:
            return [r.unique_id for r in self.all_records()]
        result = per_axis[0]
        for s in per_axis[1:]:
            result &= s
        return sorted(result)

    def _semantic_search(self, key_text: str, top_k: Optional[int] = None) -> set:
        if not self._vs.collection_exists(self.COLLECTION):
            return set()
        q_emb = self._embedding_fn(key_text)
        results = self._vs.search(self.COLLECTION, q_emb, top_k=top_k or 10)
        return {pid for pid, _, _ in results}

    # ------------------------------------------------------------------ #
    # Convenience: ObjectRecord ↔ MemoryEntry
    # ------------------------------------------------------------------ #
    def _record_to_entry(self, rec: ObjectRecord) -> MemoryEntry:
        return MemoryEntry(
            entry_id=rec.unique_id,
            text=rec.to_text(),
            payload=rec,
            semantic_keys=self._extract_semantic_tokens(rec),
            spatial_keys=[rec.spa.centroid],
            temporal_keys=list(rec.tem.timestamps),
            source_slot=MemorySlot.GM.value,
            metadata={
                "object_id": rec.unique_id,
                "description": rec.sem.description,
                "extent": rec.spa.extent,
            },
        )

    @staticmethod
    def _extract_semantic_tokens(rec: ObjectRecord) -> List[str]:
        if not rec.sem.description:
            return []
        return [w.lower() for w in rec.sem.description.split() if len(w) > 3]


# Backwards-compat: tests import `_bbox_to_centroid_extent` from this module.
__all__ = ["R4KnowledgeDatabase", "_bbox_to_centroid_extent"]
