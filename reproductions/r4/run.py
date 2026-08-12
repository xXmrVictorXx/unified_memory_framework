"""End-to-end real-data runner for the R4 reproduction on sample_video.mp4.

Pipeline:
1. QwenRunner (Qwen2.5-VL-7B, 4-bit, 4 GPUs) — VLM for object grounding
   and final answer synthesis.
2. Perceiver (DAM-v2 vits) — depth estimation per frame.
3. Frame sampling from the video at fixed FPS.
4. R4Pipeline.store(observations, segmented_objects) — ingests every
   detected object into R4KnowledgeDatabase (DedupPolicy + storage).
5. R4Pipeline.answer(question, live_perception) — two-stage reasoning.

Usage::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python -m reproductions.r4.run \\
        --video sample_video.mp4 \\
        --question "What is happening in the video?" \\
        --sample-fps 0.2 --backend neo4j --clear-storage
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
# DAM2 repo needs to be on PYTHONPATH for `from depth_anything_v2.dpt import ...`
_DAM2_PATH = PROJECT_ROOT / "depth_anything_2"
if _DAM2_PATH.exists():
    sys.path.insert(0, str(_DAM2_PATH))

from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase  # noqa: E402
from reproductions.r4.models import QwenRunner, make_vlm_decomposer  # noqa: E402
from reproductions.r4.perception import CameraIntrinsics, Perceiver  # noqa: E402
from reproductions.r4.pipeline import Observation, R4Pipeline  # noqa: E402


def _make_db(
    runner: QwenRunner,
    backend: str,
    neo4j_kwargs: dict,
    clear_storage: bool,
):
    """Build an :class:`R4KnowledgeDatabase` backed by the chosen storage."""
    from unimem.graph_storage import InMemoryGraphStorage
    from unimem.vector_storage import InMemoryVectorStorage

    if backend == "neo4j":
        from unimem.graph_storage.neo4j_backend import Neo4jGraphStorage
        gs = Neo4jGraphStorage(clear_on_init=clear_storage, **neo4j_kwargs)
        rows = gs.query("MATCH (n) RETURN count(n) AS c")
        print(f"[storage] Neo4j connected; {rows[0]['c']} nodes pre-existing")
    else:
        gs = InMemoryGraphStorage()
        print("[storage] InMemoryGraphStorage (no persistence)")

    vs = InMemoryVectorStorage()
    db = R4KnowledgeDatabase(
        embedding_fn=_make_embedding_fn(runner),
        graph_storage=gs,
        vector_storage=vs,
    )
    return db, gs


def _make_embedding_fn(runner: QwenRunner):
    """Build the SEM-axis embedder using a real sentence-transformer model.

    Uses BGE-m3 (1024-dim, multilingual, stored locally at
    ``~/.cache/modelscope/hub/models/BAAI/bge-m3``). Loaded once and cached
    on the first call.

    Falls back to a hash-based pseudo-embedder only if sentence-transformers
    or the model are unavailable (e.g. on a CI machine without the model).
    """
    model_path = "/home/eg4/.cache/modelscope/hub/models/BAAI/bge-m3"
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_path)
        # Warm up so the first real call doesn't pay the model-load cost.
        st.encode(["warmup"])
        print(f"[embedder] BGE-m3 loaded from {model_path} (dim={st.get_sentence_embedding_dimension()})")

        def embed(text: str):
            return st.encode(text, normalize_embeddings=True).tolist()

        return embed
    except (ImportError, OSError, Exception) as e:
        print(f"[embedder] WARNING: BGE-m3 unavailable ({e!r}); falling back to hash")
        import hashlib

        def embed(text: str):
            h = hashlib.sha512(text.encode("utf-8")).digest()
            dim = 64
            out = []
            i = 0
            while len(out) < dim:
                out.append((h[i % len(h)] / 127.5) - 1.0)
                i += 1
            norm = sum(x * x for x in out) ** 0.5 or 1.0
            return [x / norm for x in out]

        return embed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="sample_video.mp4")
    parser.add_argument(
        "--question", default="What is happening in the video?",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=0.2,
        help="Frame sampling rate (frames per second). 0.2 = 1 frame every 5s.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=2,
        help="Cap on R4 iterative retrieval-augmented reasoning rounds.",
    )
    parser.add_argument(
        "--depth-variant", default="vits", choices=["vits", "vitb", "vitl"],
        help="Which DAM-v2 checkpoint variant to use.",
    )
    parser.add_argument(
        "--depth-model-path", default=None,
        help="Override DAM-v2 safetensors path.",
    )
    parser.add_argument(
        "--backend", choices=["memory", "neo4j"], default="memory",
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--clear-storage", action="store_true")
    parser.add_argument(
        "--output-dir", default="tmp_r4_run",
        help="Where to save sampled frame JPEGs (for VLM grounding).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1
    video_abs = os.path.abspath(args.video)

    # Default DAM path based on variant
    if args.depth_model_path is None:
        suffix = "" if args.depth_variant == "vitl" else ""
        path = (
            f"/mnt/my_hub2/models/Depth-Anything-V2_Safetensors/"
            f"depth_anything_v2_{args.depth_variant}{suffix}.safetensors"
        )
        args.depth_model_path = path
    if not os.path.isfile(args.depth_model_path):
        print(f"ERROR: DAM weights not found: {args.depth_model_path}", file=sys.stderr)
        return 1

    # ---- 1. Load models ---------------------------------------------- #
    print("\n[model] loading QwenRunner (4-bit, 4-GPU sharding) ...")
    runner = QwenRunner()
    print(f"\n[model] loading DAM-v2 {args.depth_variant} ...")
    perceiver = Perceiver(
        qwen_runner=runner,
        depth_model_path=args.depth_model_path,
        depth_variant=args.depth_variant,
    )

    # ---- 2. Build DB ------------------------------------------------- #
    print(f"\n[storage] backend={args.backend!r}")
    db, graph_storage = _make_db(
        runner=runner,
        backend=args.backend,
        neo4j_kwargs={
            "uri": args.neo4j_uri,
            "user": args.neo4j_user,
            "password": args.neo4j_password,
        },
        clear_storage=args.clear_storage,
    )

    pipe = R4Pipeline(
        db=db,
        vlm=runner.vlm,
        decomposer=make_vlm_decomposer(runner.llm),
    )

    # ---- 3. Sample frames and store ---------------------------------- #
    import decord
    import numpy as np
    os.makedirs(args.output_dir, exist_ok=True)

    vr = decord.VideoReader(video_abs)
    n_frames = len(vr)
    fps = vr.get_avg_fps()
    step = max(1, int(round(fps / args.sample_fps)))
    frame_indices = list(range(0, n_frames, step))
    print(
        f"\n[sample] {n_frames} frames @ {fps:.1f}fps, step={step} "
        f"→ {len(frame_indices)} sampled"
    )

    print("\n[perceive] grounding + depth + 3D back-projection per frame ...")
    n_new_objects = 0
    for i, idx in enumerate(frame_indices):
        t = idx / fps
        frame = vr[idx].asnumpy()
        img_path = os.path.abspath(
            os.path.join(args.output_dir, f"frame_{i:03d}.jpg")
        )
        segs = perceiver.perceive(frame, t, img_path)
        if not segs:
            continue
        observation = Observation(timestamp=t, image=img_path, point_cloud=[])
        n_new = pipe.store(observation, segs)
        names = [s.description for s in segs]
        kept = [s.description for s, ok in zip(segs, [True] * len(segs)) if ok]
        print(
            f"  [{i:3d}] t={t:6.2f}s  detected={len(segs):2d}  new={n_new:2d}  "
            f"names={names[:5]}"
        )
        n_new_objects += n_new

    print(f"\n[perceive] done. {n_new_objects} new object records stored.")

    # ---- 4. Ask R4 --------------------------------------------------- #
    print(f"\n[answer] question: {args.question!r}")
    print(f"[answer] max_rounds={args.max_rounds}")
    t0 = time.time()
    result = pipe.answer(args.question, live_perception=None, max_rounds=args.max_rounds)
    dt = time.time() - t0

    # ---- 5. Report --------------------------------------------------- #
    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Answer               : {result.answer}")
    print(f"Storage writes       : {result.n_storage_writes}")
    print(f"Retrieval rounds     : {result.n_retrieval_rounds}")
    print(f"Used retrieval       : {result.used_retrieval}")
    print(f"Wall time            : {dt:.1f}s")
    print(f"\n--- Final retrieval context ---")
    print(result.final_context_text or "(empty)")
    print(f"\n--- Retrieval trace ---")
    for i, r in enumerate(result.retrieval_trace):
        print(f"  round {i}: k_sem={r.k_sem}, "
              f"k_spa=({r.k_spa_centroid}, r={r.k_spa_radius}), "
              f"k_t=({r.k_t_min}, {r.k_t_max}), "
              f"matches={len(r.matched_records)}")

    # ---- 6. Storage summary ------------------------------------------ #
    print("\n--- GraphStorage summary ---")
    rows = graph_storage.query("MATCH (n) RETURN count(n) AS c")
    n_nodes = rows[0]["c"]
    rows = graph_storage.query("MATCH ()-[r]->() RETURN count(r) AS c")
    n_edges = rows[0]["c"]
    print(f"Total: {n_nodes} nodes, {n_edges} edges")

    # Per-label breakdown
    rows = graph_storage.query(
        "MATCH (n) UNWIND labels(n) AS lbl "
        "RETURN lbl, count(*) AS c ORDER BY c DESC"
    )
    print("\nPer-label counts:")
    for r in rows:
        print(f"  {r['lbl']:30}  {r['c']}")

    if args.backend == "neo4j":
        print(f"\nNeo4j Browser: http://localhost:7474  "
              f"(user={args.neo4j_user}, password={args.neo4j_password})")
        print("Try:")
        print("  MATCH (o:r4_object) RETURN o.node_id, o.description, o.centroid")
        print("  MATCH (o:r4_object) WHERE o.first_seen >= 5 RETURN o")

    return 0


if __name__ == "__main__":
    sys.exit(main())
