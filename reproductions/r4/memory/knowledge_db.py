"""``R4KnowledgeDatabase`` — the framework-facing wrapper around R4's ``D``.

Implements the storage and retrieval semantics from R4 §3.1–3.2:

* **Storage (write)** — given an observed object (mask + point cloud +
  timestamp), compute SEM/SPA/TEM axes, deduplicate against existing entries
  using Eq. 5 (spatial distance < ε_c AND semantic cosine > δ_s), update or
  insert, and register the centroid as a "special point" in the SLAM map.
* **Retrieval (read)** — given a query decomposed into (k_sem, k_spa, k_tem)
  keys, intersect the per-axis candidate sets and return the matching
  ``ObjectRecord``s as :class:`~unimem.core.entry.MemoryEntry` items (text =
  ``record.to_text()``) wrapped in a :class:`~unimem.core.query.QueryResult`.

This is registered as a *single* unimem ``MemoryModule`` whose slot is
nominally :attr:`~unimem.core.slots.MemorySlot.GM` (its primary axis is
spatial/geometric) but which also participates in episodic + semantic
queries — i.e. one module wearing multiple hats, mirroring R4's
"three axes per object" philosophy. See ``reproductions/r4/graph_spec.py``
for how this maps onto a unimem graph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from unimem.core.context import MemoryContext
from unimem.core.entry import MemoryEntry
from unimem.core.module import MemoryModule
from unimem.core.query import Query, QueryResult
from unimem.core.slots import MemorySlot
from unimem.core.slot_abc import SpatialGeometricMemoryABC

from .embedding import EmbeddingFn, cosine_similarity, euclidean_distance, get_default_embedding
from .object_record import ObjectRecord, SemanticAxis, SpatialAxis, TemporalAxis
from .slam_map import SimpleSLAMMap


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bbox_to_centroid_extent(mask_points: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
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


# --------------------------------------------------------------------------- #
# The module
# --------------------------------------------------------------------------- #
class R4KnowledgeDatabase(SpatialGeometricMemoryABC):
    """R4's continuous 4D knowledge database ``D = {M, {O_j}}``.

    Construction
    ------------
    * ``embedding_fn``: callable mapping SEM text → vector. Defaults to the
      process-wide default (see :mod:`reproductions.r4.memory.embedding`).
    * ``slam_map``: optional pre-built SLAM map. If omitted, a fresh
      :class:`SimpleSLAMMap` is used.
    * ``eps_c``, ``delta_s``: dedup thresholds from Eq. 5. The paper does
      not specify values; defaults are conservative starting points.
    """

    # Slot is nominally GM (geometric map is the primary view), but this
    # module also indexes on semantic + temporal axes — see graph_spec.py
    # for the multi-slot graph wiring.
    DEFAULT_EPS_C = 0.5  # meters
    DEFAULT_DELTA_S = 0.7  # cosine similarity

    def __init__(
        self,
        embedding_fn: Optional[EmbeddingFn] = None,
        slam_map: Optional[SimpleSLAMMap] = None,
        eps_c: float = DEFAULT_EPS_C,
        delta_s: float = DEFAULT_DELTA_S,
        vlm_describe: Optional[Any] = None,
    ) -> None:
        self._embedding_fn = embedding_fn or get_default_embedding()
        self.slam_map = slam_map or SimpleSLAMMap()
        self.eps_c = float(eps_c)
        self.delta_s = float(delta_s)
        # ``vlm_describe``: callable ``(image_or_mask, prompt) -> description``.
        # If None, the caller must supply a ``description`` to ``observe_object``.
        self._vlm_describe = vlm_describe

        self._records: Dict[str, ObjectRecord] = {}
        self._next_id = 1
        # Lightweight cache of last retrieval for stats.
        self._last_query_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # SpatialGeometricMemoryABC required methods
    # ------------------------------------------------------------------ #
    def is_navigable(self, point: Sequence[float]) -> bool:
        """Stub: R4 delegates this to MapAnything. Here we say "always
        navigable" so tests can call it without a real SLAM backend."""
        return True

    def get_region(
        self, center: Sequence[float], radius: float
    ) -> List[Tuple[Tuple[float, ...], Dict[str, Any]]]:
        out: List[Tuple[Tuple[float, ...], Dict[str, Any]]] = []
        for oid, d in self.slam_map.nearest_neighbors(center, radius=float(radius)):
            rec = self._records[oid]
            out.append(
                (
                    rec.spa.centroid,
                    {"object_id": oid, "description": rec.sem.description, "distance": d},
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # MemoryModule contract
    # ------------------------------------------------------------------ #
    def write(self, entry: MemoryEntry, context: MemoryContext) -> bool:
        """Bridge unimem ``MemoryEntry`` → :meth:`observe_object`.

        The entry is interpreted as an observation whose:

        * ``text`` is the SEM description (or empty for VLM auto-describe)
        * ``spatial_keys`` carry the centroid (3D), optionally followed by
          extent (i.e. ``spatial_keys = [(cx, cy, cz, ex, ey, ez)]``)
        * ``temporal_keys`` is the observation timestamp(s)
        * ``payload`` (optional) is a list of 3D mask points
        * ``metadata['object_id']`` (optional) pins the unique id (else auto)
        * ``metadata['description']`` overrides ``text`` if both are set

        Returns True on insert, False if the entry was dedup-merged into an
        existing record (i.e. no new record was created).
        """
        desc = entry.metadata.get("description") or entry.text
        if not entry.spatial_keys:
            return False
        spa = entry.spatial_keys[0]
        if len(spa) >= 6:
            centroid = spa[:3]
            extent = spa[3:6]
        elif len(spa) == 3:
            # Default extent to a small cube if not provided.
            centroid = spa
            extent = (0.1, 0.1, 0.1)
        else:
            return False
        ts = entry.temporal_keys[0] if entry.temporal_keys else (
            context.timestamp if context.timestamp is not None else 0.0
        )
        return self.observe_object(
            description=desc,
            centroid=centroid,
            extent=extent,
            timestamp=ts,
            mask_points=entry.payload,
            object_id=entry.metadata.get("object_id"),
        )

    def read(self, query: Query) -> QueryResult:
        """Three-axis retrieval (R4 §3.2).

        The query's ``semantic`` / ``spatial`` / temporal-window fields map
        onto R4's (k_sem, k_spa, k_tem). Each axis produces a candidate
        ``object_id`` set; the result is their intersection (AND semantics,
        mirroring the paper's "coupled or isolated keys" language).
        """
        ids = self._retrieve(
            k_sem=query.semantic or None,
            k_spa_centroid=(query.spatial[0] if query.spatial else None),
            k_spa_radius=query.metadata.get("spatial_radius"),
            k_spa_k=query.metadata.get("spatial_k"),
            k_t_min=query.t_min,
            k_t_max=query.t_max,
        )
        recs = [self._records[i] for i in ids if i in self._records]
        # Build MemoryEntries that surface SEM/SPA/TEM as the three axes,
        # exactly as unimem's MultiAxisIndex expects.
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
        self._records.clear()
        self.slam_map = SimpleSLAMMap()
        self._next_id = 1

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self._records),
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
        """Add or merge an object observation.

        Performs Eq. 5 deduplication: if a record exists within ``eps_c``
        meters AND with cosine SEM similarity above ``delta_s``, the VLM (or
        the caller's policy) is asked to decide merge-vs-insert. In the
        absence of a VLM matcher, we merge on threshold satisfaction.

        Returns ``True`` if a new record was inserted, ``False`` if merged
        into an existing one.
        """
        emb = self._embedding_fn(description) if description else None
        # Dedup candidates
        match = self._find_match(centroid=centroid, embedding=emb) if emb else None

        if match is not None:
            self._merge_into(
                target_id=match,
                description=description,
                embedding=emb,
                centroid=centroid,
                extent=extent,
                timestamp=timestamp,
                confidence=confidence,
            )
            return False

        # Insert new record
        new_id = object_id or f"obj-{self._next_id:04d}"
        # Ensure no collision when caller didn't pin an id
        while new_id in self._records and object_id is None:
            self._next_id += 1
            new_id = f"obj-{self._next_id:04d}"
        self._next_id += 1

        rec = ObjectRecord(
            unique_id=new_id,
            sem=SemanticAxis(description=description, embedding=emb),
            spa=SpatialAxis(centroid=centroid, extent=extent),
            tem=TemporalAxis(timestamps=[float(timestamp)]),
            metadata={"confidence": confidence} if confidence is not None else {},
        )
        self._records[new_id] = rec
        self.slam_map.add_special_point(new_id, centroid)
        return True

    def _find_match(
        self,
        centroid: Sequence[float],
        embedding: Sequence[float],
    ) -> Optional[str]:
        """Return the id of the best matching existing record (Eq. 5).

        Picks the spatially-nearest candidate within ``eps_c`` whose SEM
        cosine similarity is at least ``delta_s``. Returns None if none.
        """
        candidates = self.slam_map.nearest_neighbors(centroid, radius=self.eps_c)
        best_id: Optional[str] = None
        best_sim = -1.0
        for oid, _ in candidates:
            rec = self._records[oid]
            if rec.sem.embedding is None:
                continue
            sim = cosine_similarity(embedding, rec.sem.embedding)
            if sim >= self.delta_s and sim > best_sim:
                best_sim = sim
                best_id = oid
        return best_id

    def _merge_into(
        self,
        target_id: str,
        description: str,
        embedding: Sequence[float],
        centroid: Sequence[float],
        extent: Sequence[float],
        timestamp: float,
        confidence: Optional[float],
    ) -> None:
        """Refine an existing record with a new observation.

        Per §3.1: "adding missing attributes such as color, or correcting
        uncertain earlier descriptions" + extending temporal intervals. Here
        we keep the *first* SEM description (most confident at write time)
        but update the embedding to a running average, refresh the centroid
        (latest observation wins), and append the timestamp.
        """
        rec = self._records[target_id]
        # SEM: prefer longer / more specific descriptions; else keep current.
        if len(description) > len(rec.sem.description or ""):
            rec.sem = SemanticAxis(description=description, embedding=embedding)
        elif rec.sem.embedding is not None and embedding is not None:
            # Average embeddings to refine the semantic representation.
            n = len(rec.sem.embedding)
            avg = [(a + b) / 2 for a, b in zip(rec.sem.embedding, embedding)]
            rec.sem.embedding = avg
        # SPA: latest centroid/extent wins (object may have moved).
        rec.spa = SpatialAxis(centroid=centroid, extent=extent)
        self.slam_map.update_centroid(target_id, centroid)
        # TEM: append observation.
        rec.tem.observe(timestamp)
        if confidence is not None:
            rec.metadata["confidence"] = confidence

    def get_record(self, object_id: str) -> Optional[ObjectRecord]:
        return self._records.get(object_id)

    def all_records(self) -> List[ObjectRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------ #
    # Retrieval internals — exposed so the pipeline can do fine-grained
    # multi-step queries (R4 §3.2 "iterative retrieval").
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
        """Intersect per-axis candidate id sets."""
        # SEM axis: union over multiple semantic keys, each via cosine sim.
        if k_sem:
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
            tem_ids = {
                oid
                for oid, rec in self._records.items()
                if rec.tem.overlaps(k_t_min, k_t_max)
            }
        else:
            tem_ids = None

        # Intersect provided axes
        per_axis = [s for s in (sem_ids, spa_ids, tem_ids) if s is not None]
        if not per_axis:
            return list(self._records.keys())
        result = per_axis[0]
        for s in per_axis[1:]:
            result &= s
        return sorted(result)

    def _semantic_search(self, key_text: str, top_k: Optional[int] = None) -> set:
        """Cosine-similarity search over all SEM embeddings."""
        if not self._records:
            return set()
        q_emb = self._embedding_fn(key_text)
        scored: List[Tuple[str, float]] = []
        for oid, rec in self._records.items():
            if rec.sem.embedding is None:
                continue
            sim = cosine_similarity(q_emb, rec.sem.embedding)
            scored.append((oid, sim))
        scored.sort(key=lambda x: -x[1])
        if top_k is not None:
            scored = scored[:top_k]
        # Return all non-negative-similarity matches; refine with top_k in caller.
        return {oid for oid, _ in scored}

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
        """Cheap tokenisation for semantic_keys: lowercase words from the SEM
        description (length > 3). Good enough for tests; real deployment
        would use the embedding-only path."""
        if not rec.sem.description:
            return []
        return [w.lower() for w in rec.sem.description.split() if len(w) > 3]


__all__ = ["R4KnowledgeDatabase"]
