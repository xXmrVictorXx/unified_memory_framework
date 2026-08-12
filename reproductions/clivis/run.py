"""End-to-end real-data runner for the CLiViS reproduction.

Wires together:
1. :class:`reproductions.clivis.models.QwenRunner` — local Qwen2.5-VL-7B
   (4-bit) acting as both LLM and VLM.
2. :func:`reproductions.clivis.video_utils.split_video_uniform` — moviepy
   segmentation.
3. :class:`reproductions.clivis.pipeline.CLiViSPipeline` — the existing
   unimem-backed facade. **No flow logic is reimplemented here**; this file
   only assembles the pieces and prints diagnostics.

Usage::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python -m reproductions.clivis.run \\
        --video sample_video.mp4 \\
        --question "What is the person doing?" \\
        --periods 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Make the project root importable when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reproductions.clivis.models import QwenRunner  # noqa: E402
from reproductions.clivis.pipeline import (  # noqa: E402
    CLiViSPipeline,
    PeriodInput,
)
from reproductions.clivis.video_utils import (  # noqa: E402
    probe_video,
    split_video_uniform,
)


def describe_period_via_vlm(
    runner: QwenRunner,
    period: PeriodInput,
    idx: int,
) -> str:
    """Ask the VLM for a 2-3 sentence summary of a single segment."""
    prompt = (
        "In 2-3 sentences, describe what happens in this video segment. "
        "Focus on: who is present, what they are doing, where they are, "
        "and any notable objects or actions."
    )
    t0 = time.time()
    resp = runner.vlm(prompt, images=[period.segment_file], max_new_tokens=120)
    dt = time.time() - t0
    preview = resp[:100] + ("..." if len(resp) > 100 else "")
    print(f"  [{idx}] {period.name} ({dt:.1f}s): {preview}")
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", default="sample_video.mp4",
        help="Path to the input video (default: sample_video.mp4)"
    )
    parser.add_argument(
        "--question", default="What is the person in the video doing?",
        help="Question to ask about the video",
    )
    parser.add_argument(
        "--periods", type=int, default=3,
        help="Number of uniform temporal segments to split the video into",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3,
        help="Cap on iterative LLM-VLM refinement rounds",
    )
    parser.add_argument(
        "--output-dir", default="tmp_clivis_run",
        help="Where to store segmented videos and logs",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Override Qwen2.5-VL model path (default: /mnt/my_hub2/models/...)",
    )
    parser.add_argument(
        "--backend", choices=["memory", "neo4j"], default="memory",
        help="GraphStorage backend. 'memory' (default) = in-process, lost on "
             "exit. 'neo4j' = persistent Neo4j at bolt://localhost:7687.",
    )
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687",
        help="Neo4j Bolt URI (only used with --backend neo4j)",
    )
    parser.add_argument(
        "--neo4j-user", default="neo4j",
        help="Neo4j username (only used with --backend neo4j)",
    )
    parser.add_argument(
        "--neo4j-password", default="password",
        help="Neo4j password (only used with --backend neo4j)",
    )
    parser.add_argument(
        "--clear-storage", action="store_true",
        help="Clear the GraphStorage before running (Neo4j: wipe all nodes; "
             "memory: no-op since it always starts empty).",
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
    runner_kwargs = {}
    if args.model_path:
        runner_kwargs["model_path"] = args.model_path
    runner = QwenRunner(**runner_kwargs)

    # ---- 2. Split video ---------------------------------------------- #
    seg_dir = os.path.join(args.output_dir, "segments")
    print(f"\n[split] {args.periods} uniform periods → {seg_dir}")
    seg_meta = split_video_uniform(video_abs, args.periods, seg_dir)
    periods: List[PeriodInput] = [
        PeriodInput(name=name, description="", segment_file=seg_path)
        for (name, seg_path, _start, _end) in seg_meta
    ]

    # ---- 3. Describe each period via VLM ----------------------------- #
    print("\n[describe] generating per-period descriptions ...")
    for i, p in enumerate(periods):
        p.description = describe_period_via_vlm(runner, p, i)

    # ---- 4. Build & run the pipeline --------------------------------- #
    print(f"\n[pipeline] question: {args.question!r}")
    print(f"[pipeline] max_rounds={args.max_rounds}")

    graph_storage = None
    if args.backend == "neo4j":
        from unimem.graph_storage.neo4j_backend import Neo4jGraphStorage
        print(f"[storage] Neo4j at {args.neo4j_uri}")
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

    pipe = CLiViSPipeline(
        llm=runner.llm,
        vlm=runner.vlm,
        max_rounds=args.max_rounds,
        graph_storage=graph_storage,
    )
    t0 = time.time()
    result = pipe.run(args.question, periods, full_video_segment=video_abs)
    dt = time.time() - t0

    # ---- 5. Report --------------------------------------------------- #
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Answer            : {result.answer}")
    print(f"Rounds executed   : {result.n_rounds}")
    print(f"Rationales stored : {result.n_rationales}")
    print(f"Pipeline wall time: {dt:.1f}s")
    print("\n--- Final scene-graph subgraph ---")
    print(result.final_subgraph_text or "(empty)")
    print("\n--- Final working memory ---")
    print(result.final_memory_text or "(empty)")
    print("\n--- Dialogue history (truncated) ---")
    for h in result.history:
        content = h["content"][:160].replace("\n", " ")
        print(f"  [{h['role']}] {content}{'...' if len(h['content']) > 160 else ''}")

    # ---- 6. Storage summary ------------------------------------------ #
    print("\n--- GraphStorage summary ---")
    rows = graph_storage.query("MATCH (n) RETURN count(n) AS c")
    n_nodes = rows[0]["c"]
    rows = graph_storage.query("MATCH ()-[r]->() RETURN count(r) AS c")
    n_edges = rows[0]["c"]
    print(f"Total: {n_nodes} nodes, {n_edges} edges")
    if args.backend == "neo4j":
        print(f"\nNeo4j Browser:  http://localhost:7474  "
              f"(user={args.neo4j_user}, password={args.neo4j_password})")
        print("Try Cypher in browser:")
        print('  MATCH (n) RETURN n                         -- all nodes')
        print('  MATCH (n:Person) RETURN n                  -- all Person nodes')
        print('  MATCH (a)-[r:PERFORMS]->(b) RETURN a, r, b -- action relations')
        print('  MATCH (n:TimeIndex) RETURN n               -- time anchors')
    return 0


if __name__ == "__main__":
    sys.exit(main())
