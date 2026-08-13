"""End-to-end real-data runner for the VideoHV reproduction.

Wires together:
1. :class:`QwenRunner` — local Qwen2.5-VL-7B (4-bit) acting as both the
   structured LLM and the clip-captioning VLM.
2. :func:`split_video_uniform` — moviepy segmentation into N uniform clips.
3. Per-clip VLM extraction — action caption + object detections for each
   clip, assembled into a :class:`VideoHVBundle`.
4. :class:`QwenVisionTools` — lets the pipeline's verification stage call
   back into the VLM on specific clips.
5. :class:`VideoHVPipeline` — the hypothesis → distinctness → verification
   → answer loop with both memories (summary + trace) backed by the chosen
   GraphStorage.

Usage::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python -m reproductions.videohv.run \\
        --video sample_video.mp4 \\
        --clips 3 --max-rounds 3 --backend neo4j
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make the project root importable when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reproductions.clivis.video_utils import probe_video, split_video_uniform  # noqa: E402
from reproductions.videohv.memory.time_verification_trace import VerificationTraceMemory  # noqa: E402
from reproductions.videohv.memory.video_summary_memory import VideoSummaryMemory  # noqa: E402
from reproductions.llm import QwenRunner, QwenVisionTools, make_structured_llm  # noqa: E402
from reproductions.videohv.pipeline import VideoHVBundle, VideoHVPipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# JSON extraction (handles ```json fences and trailing text)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Any:
    """Pull the first balanced JSON object or array out of ``text``."""
    # Strip ```json ... ``` or ``` ... ``` fences.
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Try the whole thing first (fast path).
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced { ... } or [ ... ].
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# --------------------------------------------------------------------------- #
# Per-clip VLM extraction
# --------------------------------------------------------------------------- #
def describe_clip_via_vlm(
    runner: QwenRunner,
    segment_path: str,
    idx: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Ask the VLM for an action caption + object list in a single call.

    Returns ``(action_caption, object_detections)``. On JSON parse failure
    the raw text becomes the caption and objects default to ``[]`` — the
    pipeline degrades gracefully since verification only needs the caption.
    """
    prompt = (
        "Analyze this video segment. Return ONLY a JSON object with two fields:\n"
        '  "action": a 1-2 sentence description of the main action happening\n'
        '  "objects": an array of {"label": "object_name"} for the distinct '
        "objects visible (use lowercase singular names)\n\n"
        "Example:\n"
        '{"action": "A person is painting on a canvas.", '
        '"objects": [{"label": "canvas"}, {"label": "paintbrush"}, {"label": "person"}]}'
    )
    t0 = time.time()
    resp = runner.vlm(prompt, images=[segment_path], max_new_tokens=150)
    dt = time.time() - t0
    data = _extract_json(resp)
    if isinstance(data, dict):
        action = str(data.get("action", resp[:100])).strip()
        raw_objs = data.get("objects", [])
        objects: List[Dict[str, Any]] = []
        if isinstance(raw_objs, list):
            for o in raw_objs:
                if isinstance(o, dict):
                    label = o.get("label") or o.get("name") or o.get("class")
                    if label:
                        objects.append({"label": str(label).lower().strip()})
                elif isinstance(o, str) and o.strip():
                    objects.append({"label": o.strip().lower()})
    else:
        # Fallback: use the raw response as the caption.
        action = resp.strip()
        objects = []
    preview = action[:80] + ("..." if len(action) > 80 else "")
    print(f"  [clip {idx}] ({dt:.1f}s): {preview}  |  {len(objects)} objects")
    return action, objects


# --------------------------------------------------------------------------- #
# Auto question + options generation
# --------------------------------------------------------------------------- #
def generate_question_and_options(
    runner: QwenRunner,
    action_captions: List[str],
) -> Tuple[str, List[str]]:
    """Ask the LLM to produce a 5-way multiple-choice question from the
    clip summaries. Falls back to a generic question if generation fails.
    """
    summaries_text = "\n".join(
        f"  Clip {i}: {c}" for i, c in enumerate(action_captions)
    )
    prompt = (
        "Based on these video clip summaries, create a multiple-choice "
        "question about what happened in the video.\n\n"
        f"Clip summaries:\n{summaries_text}\n\n"
        "Return ONLY a JSON object with exactly 5 options (index 0 = correct):\n"
        '{"question": "...", "options": ["A", "B", "C", "D", "E"]}'
    )
    resp = runner.llm(prompt, max_new_tokens=300)
    data = _extract_json(resp)
    if (
        isinstance(data, dict)
        and isinstance(data.get("question"), str)
        and isinstance(data.get("options"), list)
        and len(data["options"]) >= 2
    ):
        opts = [str(o) for o in data["options"][:5]]
        return data["question"].strip(), opts
    # Fallback
    print("  [gen-q] JSON parse failed, using fallback question/options")
    return (
        "What is the main activity shown in the video?",
        [
            "The person is cooking food",
            "The person is doing crafts or artwork",
            "The person is cleaning the room",
            "The person is exercising",
            "The person is working on a computer",
        ],
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", default="sample_video.mp4",
        help="Path to the input video (default: sample_video.mp4)",
    )
    parser.add_argument(
        "--question", default=None,
        help="Question to ask (if omitted, auto-generated from clip summaries)",
    )
    parser.add_argument(
        "--options", default=None,
        help="Comma-separated multiple-choice options (if omitted, auto-generated). "
             "Example: --options \"opt A,opt B,opt C,opt D,opt E\"",
    )
    parser.add_argument(
        "--clips", type=int, default=3,
        help="Number of uniform clips to split the video into (default: 3)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="Cap on hypothesis-verification refinement rounds (default: 3)",
    )
    parser.add_argument(
        "--sample-frames-per-call", type=int, default=4,
        help="Max clips sampled per vision-tools caption call (default: 4)",
    )
    parser.add_argument(
        "--no-vision-tools", action="store_true",
        help="Disable VLM-based verification; use textual-only verification "
             "over the clip summaries instead.",
    )
    parser.add_argument(
        "--output-dir", default="tmp_videohv_run",
        help="Where to store segmented videos and logs",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Override Qwen2.5-VL model path",
    )
    parser.add_argument(
        "--max-memory-per-gpu", default="7GB",
        help="Max memory per GPU for model sharding (default: 7GB). "
             "Lower this if GPUs are occupied by other processes.",
    )
    parser.add_argument(
        "--cpu-offload", default="32GB",
        help="CPU offload buffer for layers that don't fit on GPU (default: 32GB)",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU indices to use (e.g. '0,2,3'). "
             "Default: all visible GPUs.",
    )
    parser.add_argument(
        "--video-fps", type=float, default=None,
        help="Frames per second to sample from each clip for the VLM. "
             "Lower = fewer frames = faster but less detail (default: 0.5).",
    )
    parser.add_argument(
        "--backend", choices=["memory", "neo4j"], default="memory",
        help="GraphStorage backend (default: memory)",
    )
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687",
        help="Neo4j Bolt URI (only used with --backend neo4j)",
    )
    parser.add_argument(
        "--neo4j-user", default="neo4j",
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password", default="password",
        help="Neo4j password",
    )
    parser.add_argument(
        "--clear-storage", action="store_true",
        help="Clear the GraphStorage before running",
    )
    args = parser.parse_args()

    # ---- 0. Sanity --------------------------------------------------- #
    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1
    video_abs = os.path.abspath(args.video)

    meta = probe_video(video_abs)
    print(f"[video] {video_abs}")
    print(f"        {meta.duration:.1f}s @ {meta.fps:.1f}fps, "
          f"{meta.width}x{meta.height}, {meta.n_frames} frames")

    # ---- 1. Load the model ------------------------------------------- #
    print("\n[model] loading QwenRunner (this takes ~10-20s) ...")
    runner_kwargs = {
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "cpu_offload": args.cpu_offload,
    }
    if args.video_fps is not None:
        runner_kwargs["video_fps"] = args.video_fps
    if args.model_path:
        runner_kwargs["model_path"] = args.model_path
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"[model] CUDA_VISIBLE_DEVICES={args.gpus}")
    runner = QwenRunner(**runner_kwargs)

    # ---- 2. Split video ---------------------------------------------- #
    seg_dir = os.path.join(args.output_dir, "segments")
    print(f"\n[split] {args.clips} uniform clips -> {seg_dir}")
    seg_meta = split_video_uniform(video_abs, args.clips, seg_dir)

    clip_boundaries: List[Tuple[float, float]] = []
    clip_media_map: Dict[str, str] = {}
    for i, (_name, seg_path, start, end) in enumerate(seg_meta):
        clip_boundaries.append((start, end))
        clip_media_map[f"clip-{i}"] = seg_path

    # ---- 3. Describe each clip via VLM ------------------------------- #
    print("\n[describe] generating per-clip action captions + object lists ...")
    action_captions: List[str] = []
    object_detections: List[List[Dict[str, Any]]] = []
    for i, (_name, seg_path, _start, _end) in enumerate(seg_meta):
        caption, objects = describe_clip_via_vlm(runner, seg_path, i)
        action_captions.append(caption)
        object_detections.append(objects)

    print(f"\n[describe] {len(action_captions)} clips described. "
          f"Object tags: {sorted({o['label'] for objs in object_detections for o in objs})}")

    # ---- 4. Resolve question + options ------------------------------- #
    if args.question and args.options:
        question = args.question
        options = [o.strip() for o in args.options.split(",")]
        print(f"\n[qa] using user-provided question ({len(options)} options)")
    else:
        print("\n[qa] auto-generating question + options from clip summaries ...")
        question, options = generate_question_and_options(runner, action_captions)
        if args.question:
            question = args.question  # let user override just the question

    print(f"[qa] question: {question!r}")
    for i, opt in enumerate(options):
        print(f"     [{i}] {opt}")

    # ---- 5. Build storage + memories --------------------------------- #
    if args.backend == "neo4j":
        from unimem.graph_storage.neo4j_backend import Neo4jGraphStorage
        print(f"\n[storage] Neo4j at {args.neo4j_uri}")
        graph_storage = Neo4jGraphStorage(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            clear_on_init=args.clear_storage,
        )
        rows = graph_storage.query("MATCH (n) RETURN count(n) AS c")
        print(f"[storage] connected; {rows[0]['c']} nodes pre-existing")
    else:
        from unimem.graph_storage import InMemoryGraphStorage
        graph_storage = InMemoryGraphStorage()
        print("[storage] InMemoryGraphStorage (no persistence)")

    summary_memory = VideoSummaryMemory(graph_storage=graph_storage)
    trace_memory = VerificationTraceMemory(graph_storage=graph_storage)

    # ---- 6. Build vision tools (optional) ---------------------------- #
    vision_tools = None
    if not args.no_vision_tools:
        vision_tools = QwenVisionTools(runner, clip_media_map)
        print(f"[vision] QwenVisionTools wired ({len(clip_media_map)} clips registered)")
    else:
        print("[vision] disabled (--no-vision-tools); textual verification only")

    # ---- 7. Build bundle + pipeline ---------------------------------- #
    bundle = VideoHVBundle(
        action_caption_summaries=action_captions,
        object_detections_summaries=object_detections,
        clip_boundaries=clip_boundaries,
        options=options,
    )

    pipe = VideoHVPipeline(
        llm=make_structured_llm(runner),
        vision_tools=vision_tools,
        summary_memory=summary_memory,
        trace_memory=trace_memory,
        max_rounds=args.max_rounds,
        sample_frames_per_call=args.sample_frames_per_call,
    )

    # ---- 8. Run ------------------------------------------------------ #
    print(f"\n[pipeline] max_rounds={args.max_rounds}")
    t0 = time.time()
    result = pipe.run(question, bundle)
    dt = time.time() - t0

    # ---- 9. Report --------------------------------------------------- #
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Answer            : option {result.answer} -> {options[result.answer] if result.answer < len(options) else '?'}")
    print(f"Verified          : {result.verified}")
    print(f"Rounds executed   : {result.n_rounds}")
    print(f"Final clue        : {result.final_clue or '(none)'}")
    print(f"Pipeline wall time: {dt:.1f}s")

    print("\n--- Verification trace ---")
    for tr in result.trace:
        print(f"  Round {tr.round_index}:")
        print(f"    clue   : {tr.clue or '(none)'}")
        for h in tr.hypotheses:
            print(f"    hyp[{h.option}]: {h.text}")
        verdict_preview = (tr.verdict or "")[:120]
        if tr.verdict and len(tr.verdict) > 120:
            verdict_preview += "..."
        print(f"    verdict: {verdict_preview}")
        if tr.answer_choice is not None:
            print(f"    answer : option {tr.answer_choice}")

    # ---- 10. Storage summary ----------------------------------------- #
    print("\n--- GraphStorage summary ---")
    try:
        rows = graph_storage.query("MATCH (n) RETURN count(n) AS c")
        n_nodes = rows[0]["c"]
        rows = graph_storage.query("MATCH ()-[r]->() RETURN count(r) AS c")
        n_edges = rows[0]["c"]
    except (KeyError, IndexError, TypeError):
        # InMemoryGraphStorage may not support count() aggregation.
        all_nodes = graph_storage.query("MATCH (n) RETURN n")
        n_nodes = len(all_nodes) if all_nodes else 0
        all_edges = graph_storage.query("MATCH ()-[r]->() RETURN r")
        n_edges = len(all_edges) if all_edges else 0
    print(f"Total: {n_nodes} nodes, {n_edges} edges")

    # Clip-level detail
    rows = graph_storage.query(
        "MATCH (n:VideoClip) RETURN n.clip_index AS idx, n.start_t AS s, n.end_t AS e "
        "ORDER BY n.clip_index"
    )
    if rows:
        print(f"VideoClip nodes: {len(rows)}")
        for r in rows[:5]:
            print(f"  clip-{r.get('idx')}: [{r.get('s'):.1f}, {r.get('e'):.1f}]")

    # Trace nodes
    trace_rows = graph_storage.query("MATCH (n:Trace) RETURN n")
    if trace_rows:
        print(f"Trace nodes: {len(trace_rows)}")

    if args.backend == "neo4j":
        print(f"\nNeo4j Browser:  http://localhost:7474  "
              f"(user={args.neo4j_user}, password={args.neo4j_password})")
        print("Try Cypher in browser:")
        print('  MATCH (n:VideoClip) RETURN n               -- clip summaries')
        print('  MATCH (n:Trace) RETURN n                   -- verification trace')
        print('  MATCH (n:TimeIndex) RETURN n               -- time anchors')
        print('  MATCH (a:VideoClip)-[:AT_TIME]->(t) RETURN a, t  -- clip timelines')
    return 0


if __name__ == "__main__":
    sys.exit(main())
