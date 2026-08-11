"""Video segmentation utilities for the CLiViS reproduction.

Splits an input video into N uniform mp4 segments named after their
``hh:mm:ss-hh:mm:ss`` time ranges — the format CLiViS's
:class:`~reproductions.clivis.pipeline.PeriodInput` expects.

Pure stdlib + moviepy + opencv. No imports from ``reproduce/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class VideoMeta:
    """Lightweight metadata extracted from a video file."""

    path: str
    duration: float
    fps: float
    n_frames: int
    width: int
    height: int


def probe_video(path: str) -> VideoMeta:
    """Read duration / fps / dimensions from ``path`` via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) for {path}")
    return VideoMeta(
        path=path,
        duration=n_frames / fps,
        fps=fps,
        n_frames=n_frames,
        width=width,
        height=height,
    )


def seconds_to_hhmmss(s: float) -> str:
    total = int(s)
    h = total // 3600
    m = (total % 3600) // 60
    sec = total % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def hhmmss_to_seconds(text: str) -> float:
    parts = [int(x) for x in text.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    else:
        raise ValueError(f"Bad time format: {text!r}")
    return h * 3600 + m * 60 + s


def split_video_uniform(
    video_path: str,
    n_periods: int,
    output_dir: str,
    clear_existing: bool = True,
) -> List[tuple]:
    """Split ``video_path`` into ``n_periods`` equal-length mp4 segments.

    Returns ``[(period_name, segment_path, start_sec, end_sec), ...]`` where
    ``period_name`` is ``"hh:mm:ss-hh:mm:ss"``.
    """
    from moviepy.video.io.VideoFileClip import VideoFileClip

    meta = probe_video(video_path)
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    os.makedirs(output_dir, exist_ok=True)
    if clear_existing:
        for f in os.listdir(output_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(output_dir, f))

    seg_len = meta.duration / n_periods
    clip = VideoFileClip(video_path, audio=False)
    out: List[tuple] = []
    try:
        for i in range(n_periods):
            start, end = i * seg_len, (i + 1) * seg_len
            # The final segment eats any rounding remainder.
            if i == n_periods - 1:
                end = meta.duration
            name = f"{seconds_to_hhmmss(start)}-{seconds_to_hhmmss(end)}"
            seg_path = os.path.abspath(
                os.path.join(output_dir, f"period_{i:02d}.mp4")
            )
            sub = clip.subclipped(start, end)
            sub.write_videofile(
                seg_path,
                codec="libx264",
                audio=False,
                preset="ultrafast",
                logger=None,
            )
            sub.close()
            out.append((name, seg_path, start, end))
    finally:
        clip.close()
    return out


__all__ = [
    "VideoMeta",
    "probe_video",
    "seconds_to_hhmmss",
    "hhmmss_to_seconds",
    "split_video_uniform",
]
