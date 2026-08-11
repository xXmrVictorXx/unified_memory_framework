"""CLiViS inference pipeline skeleton.

Reproduces ``reproduce/CLiViS/clivis/pipeline/time_inference_od_neo4j.py:inference``.

Phase 1 — Initialisation:
  * Split video into temporal periods (caller-supplied here)
  * Build initial Cognitive Map via LLM extraction on each period's description
  * Mark key entities from the question

Phase 2 — Iterative LLM-VLM refinement (up to ``max_rounds``):
  * Read nav info + clue subgraph + rationale memory
  * LLM generates next instruction (or final answer)
  * VLM executes the instruction on the relevant video segment
  * Update working memory (rationale extraction)
  * Update scene graph (entity/relation/action extraction)

Phase 3 — Final fallback answer.

All LLM/VLM calls are injectable. Tests script the conversation via
``MockLLM`` pattern_map / response queue.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from unimem.core.context import MemoryContext

from .memory.navigation_graph import NavigationGraph
from .memory.relation_graph import RelationGraph
from .memory.time_working_memory import TimeWorkingMemory

LLMFn = Callable[[str], str]
VLMFn = Callable[[str, Optional[List[Any]]], str]


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class PeriodInput:
    """Caller-supplied video segment descriptor."""

    name: str  # e.g. "00:00:00-00:00:30"
    description: str  # text summary of what happens in this segment
    segment_file: Optional[str] = None  # path to clipped video file


@dataclass
class CLiViSResult:
    answer: str
    n_rounds: int
    n_rationales: int
    final_subgraph_text: str
    final_memory_text: str
    history: List[Dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class CLiViSPipeline:
    """CLiViS iterative refinement loop with injectable LLM/VLM."""

    FINAL_ANSWER_TOKEN = "[final]"
    INSTRUCTION_TOKEN = "[instruction]"

    def __init__(
        self,
        llm: LLMFn,
        vlm: VLMFn,
        max_rounds: int = 15,
    ) -> None:
        self.llm = llm
        self.vlm = vlm
        self.max_rounds = int(max_rounds)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        periods: List[PeriodInput],
        full_video_segment: Optional[str] = None,
    ) -> CLiViSResult:
        # ---- Phase 1: Initialise Cognitive Map ----
        nav = NavigationGraph(
            period_description_dict={p.name: p.description for p in periods}
        )
        rel = RelationGraph()
        wm = TimeWorkingMemory(question=question, llm_extractor=self.llm)
        for p in periods:
            if p.segment_file:
                nav.register_video_segment(p.name, p.segment_file)
        self._initialise_cognitive_map(nav, rel, periods, question)

        # ---- VLM full-video evidence (one-shot before the loop) ----
        full_video_segment = full_video_segment or (
            periods[0].segment_file if periods else None
        )
        initial_response = self.vlm(
            self._build_initial_perception_prompt(question), [full_video_segment]
        )
        wm.update_history_msg("[initial perception]", initial_response)

        # ---- Phase 2: iterative LLM-VLM refinement ----
        for round_idx in range(self.max_rounds):
            nav_info = nav.output_periods_info()
            subgraph = rel.extract_subgraph_by_nodes(
                list(self._marked_entity_names(rel, wm, question))
            )
            subgraph_text = rel.format_subgraph(subgraph)
            memory_text = wm.output_memory_info()

            llm_output = self.llm(
                self._build_instruction_prompt(
                    question=question,
                    nav_info=nav_info,
                    subgraph_text=subgraph_text,
                    memory_text=memory_text,
                    history=wm.history_messages,
                )
            )

            if self.FINAL_ANSWER_TOKEN in llm_output:
                answer = llm_output.replace(self.FINAL_ANSWER_TOKEN, "").strip()
                return CLiViSResult(
                    answer=answer,
                    n_rounds=round_idx,
                    n_rationales=wm.get_rationale_count(),
                    final_subgraph_text=subgraph_text,
                    final_memory_text=memory_text,
                    history=list(wm.history_messages),
                )

            # Parse instruction (period, instruction_text)
            period, instruction_text = self._parse_instruction(llm_output)
            if not period or not instruction_text:
                continue
            seg = nav.video_segments_to_files.get(period)
            vlm_response = self.vlm(
                self._build_vlm_prompt(instruction_text, question), [seg]
            )
            # ---- Phase 2.5: Write back to memory ----
            wm.update_history_msg(instruction_text, vlm_response)
            wm.extract_and_update_rationale_list(period)
            self._update_cognitive_map_from_vlm(nav, rel, vlm_response, period, question)

        # ---- Phase 3: fallback final answer ----
        nav_info = nav.output_periods_info()
        subgraph = rel.extract_subgraph_by_nodes(
            list(self._marked_entity_names(rel, wm, question))
        )
        subgraph_text = rel.format_subgraph(subgraph)
        memory_text = wm.output_memory_info()
        fallback = self.llm(
            self._build_fallback_prompt(
                question=question,
                nav_info=nav_info,
                subgraph_text=subgraph_text,
                memory_text=memory_text,
            )
        )
        return CLiViSResult(
            answer=fallback,
            n_rounds=self.max_rounds,
            n_rationales=wm.get_rationale_count(),
            final_subgraph_text=subgraph_text,
            final_memory_text=memory_text,
            history=list(wm.history_messages),
        )

    # ------------------------------------------------------------------ #
    # LLM-driven Cognitive Map init / update (skeleton)
    # ------------------------------------------------------------------ #
    def _initialise_cognitive_map(
        self,
        nav: NavigationGraph,
        rel: RelationGraph,
        periods: List[PeriodInput],
        question: str,
    ) -> None:
        """Skeleton version of CLiViS's init_persons_and_areas / init_obj /
        init_obj_rel / init_action.

        The real method issues 4 LLM calls; here we issue one combined call
        and parse JSON. Tests can script the response via MockLLM patterns.
        """
        joined = "\n".join(f"{p.name}: {p.description}" for p in periods)
        prompt = (
            "Extract structured scene-graph information from the following video periods.\n"
            "Return JSON with keys: persons (list of {name, info}), "
            "areas (list of {name, info, time_range}), "
            "objects (list of {name, period}), "
            "relations (list of {source, type, target, time_range}), "
            "actions (list of {name, info, time_range, agent, patient?, "
            "instrument?, source?, target?}).\n"
            f"Question: {question}\n"
            f"Periods:\n{joined}"
        )
        raw = self.llm(prompt)
        parsed = _safe_json_extract(raw)
        if not parsed:
            return
        for p in parsed.get("persons", []):
            nav.add_persons([{"name": p.get("name"), "info": p.get("info", "")}])
            if p.get("name"):
                rel.add_person(p["name"], p.get("info", ""))
        for a in parsed.get("areas", []):
            nav.add_areas([a])
            rel.add_area(a.get("name", ""), a.get("time_range", ""), a.get("info", ""))
        for o in parsed.get("objects", []):
            name = o.get("name")
            period = o.get("period", "")
            if name and period:
                nav.add_objs([name], period)
                rel.add_update_objects(name, period, "")
        for r in parsed.get("relations", []):
            rel.add_relation(
                r.get("source", ""), r.get("target", ""),
                r.get("type", "RELATED"), r.get("info", ""),
                r.get("time_range", ""), None,
            )
        for a in parsed.get("actions", []):
            rel.add_action(
                action_name=a.get("name", ""),
                action_info=a.get("info", ""),
                time_range=a.get("time_range", ""),
                node_agent_name=a.get("agent"),
                node_patient_name=a.get("patient"),
                node_instrument_name=a.get("instrument"),
                node_source_name=a.get("source"),
                node_target_name=a.get("target"),
            )

    def _update_cognitive_map_from_vlm(
        self,
        nav: NavigationGraph,
        rel: RelationGraph,
        vlm_response: str,
        period: str,
        question: str,
    ) -> None:
        """Skeleton version of CLiViS's update_obj_rel_act.

        Asks the LLM to extract any *new* entities/relations/actions from the
        latest VLM response. Same JSON shape as ``_initialise_cognitive_map``.
        """
        prompt = (
            "Extract any NEW scene-graph information from the following VLM response. "
            "Return JSON with keys: objects, relations, actions (same shape as init).\n"
            f"Period: {period}\n"
            f"Question: {question}\n"
            f"VLM response: {vlm_response}"
        )
        raw = self.llm(prompt)
        parsed = _safe_json_extract(raw)
        if not parsed:
            return
        for o in parsed.get("objects", []):
            name = o.get("name")
            if name:
                nav.add_objs([name], period)
                rel.add_update_objects(name, period, o.get("info", ""))
        for r in parsed.get("relations", []):
            rel.add_relation(
                r.get("source", ""), r.get("target", ""),
                r.get("type", "RELATED"), r.get("info", ""),
                r.get("time_range", period), None,
            )
        for a in parsed.get("actions", []):
            rel.add_action(
                action_name=a.get("name", ""),
                action_info=a.get("info", ""),
                time_range=a.get("time_range", period),
                node_agent_name=a.get("agent"),
                node_patient_name=a.get("patient"),
                node_instrument_name=a.get("instrument"),
                node_source_name=a.get("source"),
                node_target_name=a.get("target"),
            )

    # ------------------------------------------------------------------ #
    # Prompts & parsers
    # ------------------------------------------------------------------ #
    def _build_initial_perception_prompt(self, question: str) -> str:
        return (
            "Watch this video and extract any initial evidence relevant to the question.\n"
            f"Question: {question}"
        )

    def _build_instruction_prompt(
        self,
        question: str,
        nav_info: str,
        subgraph_text: str,
        memory_text: str,
        history: List[Dict[str, str]],
    ) -> str:
        history_text = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in history[-6:]
        )
        return (
            "Decide the next action.\n"
            f"Question: {question}\n\n"
            f"Navigation:\n{nav_info}\n\n"
            f"Scene subgraph:\n{subgraph_text}\n\n"
            f"Working memory:\n{memory_text}\n\n"
            f"Recent history:\n{history_text}\n\n"
            "Reply with either:\n"
            f"  {self.FINAL_ANSWER_TOKEN}<answer>  (if you can answer now)\n"
            f"  {self.INSTRUCTION_TOKEN}<period>|<instruction>  (to query a segment)"
        )

    def _build_vlm_prompt(self, instruction: str, question: str) -> str:
        return (
            f"Answer the following instruction about this video segment.\n"
            f"Question context: {question}\n"
            f"Instruction: {instruction}"
        )

    def _build_fallback_prompt(
        self, question: str, nav_info: str, subgraph_text: str, memory_text: str
    ) -> str:
        return (
            "You have exhausted your reasoning rounds. Give the best answer from memory.\n"
            f"Question: {question}\n\n"
            f"Navigation:\n{nav_info}\n\n"
            f"Subgraph:\n{subgraph_text}\n\n"
            f"Working memory:\n{memory_text}"
        )

    def _parse_instruction(self, llm_output: str) -> tuple[str, str]:
        """Parse ``[instruction]<period>|<text>`` (or fallback JSON / freeform)."""
        if self.INSTRUCTION_TOKEN not in llm_output:
            return "", ""
        body = llm_output.split(self.INSTRUCTION_TOKEN, 1)[1].strip()
        # Try pipe-separated
        if "|" in body:
            period, instr = body.split("|", 1)
            return period.strip(), instr.strip()
        return "", body

    def _marked_entity_names(
        self, rel: RelationGraph, wm: TimeWorkingMemory, question: str
    ) -> List[str]:
        """Key entities to centre the subgraph on.

        Skeleton heuristic: take all node names mentioned in the question,
        plus the most recent rationale's area/object.
        """
        names = set()
        for node_name in rel.all_node_names():
            if node_name.lower() in question.lower():
                names.add(node_name)
        if wm.rationale_list:
            last = wm.rationale_list[-1]
            if last.related_area:
                names.add(last.related_area)
            if last.related_obj:
                names.add(last.related_obj)
        return list(names)


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Same robust JSON extraction used by TimeWorkingMemory."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


__all__ = ["CLiViSPipeline", "CLiViSResult", "PeriodInput"]
