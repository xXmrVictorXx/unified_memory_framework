"""R4 perception module: object grounding + depth estimation + 2D→3D back-projection.

Pipeline:
1. Qwen2.5-VL grounds objects in a frame (returns name + bbox per object).
2. Depth-Anything-V2 (vits) produces a per-pixel depth map.
3. For each bbox, sample pixels inside, back-project to 3D camera coords
   via a pinhole camera model, compute centroid + extent (bbox-style in 3D).
4. Returns ``List[SegmentedObject]`` ready for :meth:`R4Pipeline.store`.

Camera model
------------
A pinhole camera with assumed intrinsics (no calibration available for the
sample video). The default is ``fx=fy=500, cx=W/2, cy=H/2`` — a reasonable
guess for a 640x360 third-person camera. Override via ``camera_intrinsics``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

# Make the cloned DAM2 repo importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DAM2_PATH = os.path.join(_PROJECT_ROOT, "depth_anything_2")
if _DAM2_PATH not in sys.path:
    sys.path.insert(0, _DAM2_PATH)


@dataclass
class CameraIntrinsics:
    """Simple pinhole intrinsics for back-projection."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def default_for(cls, width: int, height: int) -> "CameraIntrinsics":
        # Common guess for uncalibrated cameras: focal length ~ image diagonal/2
        return cls(fx=500.0, fy=500.0, cx=width / 2, cy=height / 2)


class Perceiver:
    """Object grounding + depth estimation + 3D back-projection.

    Construction
    ------------
    qwen_runner:
        A :class:`reproductions.clivis.models.QwenRunner` (or any object
        exposing ``vlm(prompt, images=[path]) -> str``).
    depth_model_path:
        Path to a DAM-v2 safetensors checkpoint (vits/vitb/vitl).
    depth_variant:
        ``"vits"`` | ``"vitb"`` | ``"vitl"`` — must match the checkpoint.
    camera_intrinsics:
        Override the default pinhole guess.
    depth_input_size:
        DAM patch size (518 for the ViT-S/B/L defaults).
    """

    # DAM-v2 model config table
    _DAM_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 768, 1024]},
    }

    def __init__(
        self,
        qwen_runner: Any,
        depth_model_path: str,
        depth_variant: str = "vits",
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        depth_input_size: int = 518,
        max_objects_per_frame: int = 8,
    ) -> None:
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        from safetensors.torch import load_file

        if depth_variant not in self._DAM_CONFIGS:
            raise ValueError(
                f"depth_variant must be one of {list(self._DAM_CONFIGS)}, "
                f"got {depth_variant!r}"
            )

        self._qwen = qwen_runner
        self._depth_input_size = int(depth_input_size)
        self._max_objects = int(max_objects_per_frame)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load DAM-v2
        cfg = self._DAM_CONFIGS[depth_variant]
        self._dam = DepthAnythingV2(**cfg)
        sd = load_file(depth_model_path)
        self._dam.load_state_dict(sd)
        self._dam.to(self._device).eval()
        print(f"[Perceiver] DAM-v2 {depth_variant} loaded from {depth_model_path}")

        # Default intrinsics tied to no specific frame size; updated per call
        self._default_intrinsics = camera_intrinsics

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def perceive(
        self,
        frame_np: Any,
        timestamp: float,
        temp_image_path: str,
    ) -> List[Any]:
        """Perceive objects in a single frame.

        Parameters
        ----------
        frame_np:
            ``(H, W, 3)`` uint8 numpy array (BGR or RGB — both fine for
            depth estimation; VLM sees the saved file).
        timestamp:
            Frame's timestamp in seconds. Stored on each ``SegmentedObject``.
        temp_image_path:
            Where to save the frame as a JPEG so Qwen2.5-VL can load it
            via ``file://``. Caller is responsible for cleanup.

        Returns
        -------
        ``List[SegmentedObject]`` with ``mask_points`` (3D), ``timestamp``,
        ``description``, and ``object_id_hint`` filled in. Objects whose
        grounding failed or whose bbox had zero area are skipped.
        """
        from PIL import Image
        import numpy as np

        # 1. Depth estimation (RGB or BGR both work — DAM treats it as image)
        h, w = frame_np.shape[:2]
        depth = self._infer_depth(frame_np)

        # 2. Object grounding via VLM
        img = Image.fromarray(frame_np.astype("uint8"))
        img.save(temp_image_path, quality=90)
        detections = self._ground_objects(temp_image_path)
        if not detections:
            return []

        # 3. 3D back-projection for each bbox
        intrinsics = self._default_intrinsics or CameraIntrinsics.default_for(w, h)
        segmented: List[Any] = []
        from reproductions.r4.pipeline import SegmentedObject  # local import

        for det in detections[: self._max_objects]:
            name = det.get("name", "").strip()
            bbox = det.get("bbox") or det.get("box")
            if not name or not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            mask_points = self._bbox_to_3d_points(
                x1, y1, x2, y2, depth, intrinsics, stride=4
            )
            if not mask_points:
                continue
            segmented.append(SegmentedObject(
                mask_points=mask_points,
                timestamp=float(timestamp),
                image_or_mask=temp_image_path,
                description=name,
                object_id_hint=name,  # R4 DB uses description text for dedup,
                                       # object_id_hint only pins id on first sight
            ))
        return segmented

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _infer_depth(self, frame_np: Any) -> Any:
        """Run DAM-v2 on a frame, return ``(H, W)`` float32 depth map."""
        import torch
        with torch.no_grad():
            depth = self._dam.infer_image(frame_np, input_size=self._depth_input_size)
        return depth

    def _ground_objects(self, image_path: str) -> List[dict]:
        """Ask Qwen2.5-VL to list objects with their bboxes in ``image_path``.

        Returns a list of ``{"name": str, "bbox": [x1, y1, x2, y2]}`` dicts.
        On parse failure, returns an empty list.
        """
        import json
        import re
        from reproductions.clivis.pipeline import _safe_json_extract

        prompt = (
            "List every distinct object visible in this image. "
            "For each object output JSON: "
            '{"name": "<short noun>", "bbox": [x1, y1, x2, y2]}. '
            "Coordinates are in image pixels (origin top-left). "
            "Output a JSON array of these objects, no other text. "
            "Limit to the 8 most salient objects."
        )
        raw = self._qwen.vlm(prompt, images=[image_path], max_new_tokens=400)
        if not raw:
            return []
        # The VLM may wrap the array in a top-level object — accept either.
        # Try plain JSON list first.
        try:
            parsed = json.loads(raw.strip())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try fence-stripped or brace-extracted
        extracted = _safe_json_extract(raw)
        if isinstance(extracted, dict) and "objects" in extracted:
            return list(extracted["objects"])
        if isinstance(extracted, list):
            return extracted
        return []

    @staticmethod
    def _bbox_to_3d_points(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        depth: Any,
        K: CameraIntrinsics,
        stride: int = 4,
    ) -> List[Tuple[float, float, float]]:
        """Back-project bbox pixels (sub-sampled by ``stride``) to 3D.

        Returns a list of ``(X, Y, Z)`` tuples in camera coordinates.
        Uses the pinhole model: ``X = (u - cx) * d / fx``, etc.
        """
        import numpy as np

        ys = list(range(y1, y2, stride))
        xs = list(range(x1, x2, stride))
        if not ys or not xs:
            return []
        pts: List[Tuple[float, float, float]] = []
        for v in ys:
            for u in xs:
                d = float(depth[v, u])
                if d <= 0 or not (0.1 < d < 100.0):
                    continue  # skip invalid / outlier depths
                X = (u - K.cx) * d / K.fx
                Y = (v - K.cy) * d / K.fy
                Z = d
                pts.append((X, Y, Z))
        return pts


__all__ = ["CameraIntrinsics", "Perceiver"]
