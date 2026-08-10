"""R4 retrieval-augmented reasoning loop (pipeline skeleton).

Faithful to R4 §3.2:

1. **Self-estimated answerability** — VLM attempts the query from live perception.
2. **Query decomposition** — VLM breaks the query into ``(k_sem, k_spa, k_tem)``.
3. **Three-axis retrieval** — fetch matching ObjectRecords from the DB.
4. **Augmented reasoning** — VLM produces the final answer from query + context.
5. **Iterative refinement** — the answer can become the next query input.

The VLM is an injectable callable. Tests use :class:`~reproductions._common.mocks.MockVLM`;
production use wires a real Gemma3-4B-IT (or any ``(prompt, images) -> str``).

The pipeline never touches model weights or sensors directly. "Live
perception" is just a ``(image, point_cloud, timestamp)`` tuple the caller
provides; segmenting it into per-object masks is the caller's job (the paper
uses SAM2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from unimem.core.context import MemoryContext
from unimem.core.query import Query, QueryBuilder

from .memory.knowledge_db import R4KnowledgeDatabase
from .memory.object_record import ObjectRecord


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
VLMFn = Callable[..., str]
DecompFn = Callable[[str, Any], Dict[str, Any]]


@dataclass
class Observation:
    """A single timestep's perceptual input (caller-supplied)."""

    timestamp: float
    image: Any = None  # path / array / PIL.Image — opaque to the pipeline
    point_cloud: List[Tuple[float, float, float]] = field(default_factory=list)
    camera_pose: Optional[Tuple[float, ...]] = None


@dataclass
class SegmentedObject:
    """An object extracted from an observation (caller-supplied, via SAM2)."""

    mask_points: List[Tuple[float, float, float]]
    timestamp: float
    image_or_mask: Any = None  # whatever the VLM describer expects
    description: Optional[str] = None  # caller can pre-fill; else VLM is called
    object_id_hint: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class RetrievalResult:
    """One round of retrieval — kept as a dataclass for inspection."""

    k_sem: List[str]
    k_spa_centroid: Optional[Tuple[float, ...]]
    k_spa_radius: Optional[float]
    k_t_min: Optional[float]
    k_t_max: Optional[float]
    matched_records: List[ObjectRecord]


@dataclass
class R4PipelineResult:
    """End-to-end pipeline outcome."""

    answer: str
    n_storage_writes: int
    n_retrieval_rounds: int
    retrieval_trace: List[RetrievalResult]
    final_context_text: str
    used_retrieval: bool


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class R4Pipeline:
    """R4's two-stage reasoning loop with optional storage pre-pass.

    Parameters
    ----------
    db:
        The :class:`R4KnowledgeDatabase` to read/write.
    vlm:
        Callable ``(prompt: str, images: list = None, **kw) -> str`` for
        description generation and final answer synthesis.
    decomposer:
        Callable ``(query: str, live_perception: Any) -> dict`` that returns
        ``{"k_sem": [...], "k_spa_centroid": (x,y,z)|None, "k_spa_radius": float|None,
           "k_t_min": float|None, "k_t_max": float|None}``. If ``None``, a
        simple keyword-based heuristic decomposer is used.
    """

    CONFIDENCE_TOKEN = "[confident]"
    NO_CONFIDENCE_TOKEN = "[need-retrieval]"

    def __init__(
        self,
        db: R4KnowledgeDatabase,
        vlm: VLMFn,
        decomposer: Optional[DecompFn] = None,
    ) -> None:
        self.db = db
        self.vlm = vlm
        self.decomposer = decomposer or _heuristic_decomposer

    # ------------------------------------------------------------------ #
    # Storage pass
    # ------------------------------------------------------------------ #
    def store(
        self,
        observation: Observation,
        segmented_objects: Sequence[SegmentedObject],
        context: Optional[MemoryContext] = None,
    ) -> int:
        """Ingest segmented objects into the knowledge database.

        Returns the number of *new* records created (merges don't count).
        """
        ctx = context or MemoryContext(timestamp=observation.timestamp)
        new_count = 0
        for seg in segmented_objects:
            description = seg.description
            if description is None:
                # VLM auto-describe via the self-prompt from §3.1.
                description = self.vlm(
                    "Provide a concise semantic object description of the given single instance.",
                    images=[seg.image_or_mask] if seg.image_or_mask is not None else None,
                )
            # Compute centroid/extent from mask points (Eqs. 1 & 2).
            from .memory.knowledge_db import _bbox_to_centroid_extent

            centroid, extent = _bbox_to_centroid_extent(seg.mask_points)
            inserted = self.db.observe_object(
                description=description,
                centroid=centroid,
                extent=extent,
                timestamp=seg.timestamp,
                mask_points=seg.mask_points,
                object_id=seg.object_id_hint,
                confidence=seg.confidence,
            )
            if inserted:
                new_count += 1
        # Record the agent's pose so the SLAM map has a trajectory.
        if observation.camera_pose is not None:
            self.db.slam_map.append_ego_pose(observation.timestamp, observation.camera_pose)
        return new_count

    # ------------------------------------------------------------------ #
    # Retrieval-augmented reasoning loop (§3.2)
    # ------------------------------------------------------------------ #
    def answer(
        self,
        question: str,
        live_perception: Optional[Observation] = None,
        max_rounds: int = 3,
    ) -> R4PipelineResult:
        """Two-stage answer loop.

        Stage 1: VLM tries to answer from live perception alone.
        Stage 2: iteratively decompose → retrieve → augment → re-answer.
        """
        # ---- Stage 1: self-estimated answerability ----
        stage1_prompt = self._build_stage1_prompt(question, live_perception)
        stage1_out = self.vlm(stage1_prompt)
        if self.CONFIDENCE_TOKEN in stage1_out:
            return R4PipelineResult(
                answer=stage1_out.replace(self.CONFIDENCE_TOKEN, "").strip(),
                n_storage_writes=0,
                n_retrieval_rounds=0,
                retrieval_trace=[],
                final_context_text="(answered from live perception)",
                used_retrieval=False,
            )

        # ---- Stage 2: retrieval-augmented reasoning ----
        trace: List[RetrievalResult] = []
        context_text = ""
        last_answer = stage1_out.replace(self.NO_CONFIDENCE_TOKEN, "").strip()
        for round_idx in range(max_rounds):
            keys = self.decomposer(question, live_perception)
            retrieved = self._retrieve(keys)
            trace.append(retrieved)
            context_text = self._serialise_context(retrieved.matched_records)
            prompt = self._build_stage2_prompt(question, context_text, last_answer)
            last_answer = self.vlm(prompt)
            if self.CONFIDENCE_TOKEN in last_answer:
                last_answer = last_answer.replace(self.CONFIDENCE_TOKEN, "").strip()
                break

        return R4PipelineResult(
            answer=last_answer,
            n_storage_writes=0,
            n_retrieval_rounds=len(trace),
            retrieval_trace=trace,
            final_context_text=context_text,
            used_retrieval=True,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _retrieve(self, keys: Dict[str, Any]) -> RetrievalResult:
        k_sem = list(keys.get("k_sem") or [])
        k_spa_centroid = keys.get("k_spa_centroid")
        k_spa_radius = keys.get("k_spa_radius")
        k_t_min = keys.get("k_t_min")
        k_t_max = keys.get("k_t_max")
        ids = self.db._retrieve(
            k_sem=k_sem or None,
            k_spa_centroid=k_spa_centroid,
            k_spa_radius=k_spa_radius,
            k_t_min=k_t_min,
            k_t_max=k_t_max,
        )
        recs = [self.db.get_record(i) for i in ids]
        recs = [r for r in recs if r is not None]
        return RetrievalResult(
            k_sem=k_sem,
            k_spa_centroid=tuple(k_spa_centroid) if k_spa_centroid else None,
            k_spa_radius=k_spa_radius,
            k_t_min=k_t_min,
            k_t_max=k_t_max,
            matched_records=recs,
        )

    @staticmethod
    def _serialise_context(records: List[ObjectRecord]) -> str:
        if not records:
            return "(no matching records in 4D knowledge database)"
        lines = [r.to_text() for r in records]
        return "\n".join(lines)

    def _build_stage1_prompt(self, question: str, live: Optional[Observation]) -> str:
        lp = "(no live perception provided)" if live is None else "(live perception available)"
        return (
            "Attempt to answer the following question from live perception alone.\n"
            f"{lp}\n"
            f"Question: {question}\n"
            "If you can answer confidently, prefix your answer with [confident]. "
            "If you need to retrieve from memory, prefix with [need-retrieval]."
        )

    def _build_stage2_prompt(self, question: str, context: str, prior_answer: str) -> str:
        return (
            "Answer the question using the retrieved 4D context below.\n"
            f"Question: {question}\n"
            f"Retrieved context:\n{context}\n"
            f"Prior attempt: {prior_answer}\n"
            "If your new answer is final, prefix with [confident]; "
            "otherwise prefix with [need-retrieval] to request another round."
        )


# --------------------------------------------------------------------------- #
# Default heuristic query decomposer
# --------------------------------------------------------------------------- #
def _heuristic_decomposer(query: str, live_perception: Any) -> Dict[str, Any]:
    """Very simple keyword-based query decomposition.

    The paper delegates this to the VLM (§3.2). Here we extract:

    * ``k_sem`` — significant nouns in the query (length > 4)
    * ``k_t_min`` / ``k_t_max`` — recognised temporal phrases ("X seconds ago")
    * ``k_spa_centroid`` / ``k_spa_radius`` — left None; the real decomposer
      would parse phrases like "10 m ahead".

    Production wiring: pass a real VLM-backed decomposer to
    :class:`R4Pipeline`.
    """
    import re

    k_sem = [w.strip(".,?!").lower() for w in query.split() if len(w) > 4]

    k_t_min: Optional[float] = None
    k_t_max: Optional[float] = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*seconds?\s*ago", query, re.IGNORECASE)
    if m:
        # "5 seconds ago" → look back 5s from "now" (caller-supplied)
        # but we don't know "now" here — leave as offset; caller can adjust.
        k_t_min = -float(m.group(1))  # negative sentinel; pipeline may rebase

    return {
        "k_sem": k_sem,
        "k_spa_centroid": None,
        "k_spa_radius": None,
        "k_t_min": k_t_min,
        "k_t_max": k_t_max,
    }


__all__ = [
    "Observation",
    "SegmentedObject",
    "RetrievalResult",
    "R4PipelineResult",
    "R4Pipeline",
    "VLMFn",
    "DecompFn",
]
