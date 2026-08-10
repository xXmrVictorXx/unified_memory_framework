"""Per-step execution context passed to memory modules.

``MemoryContext`` bundles the bits of "current agent state" that are useful for
write gating (event triggers), time-stamping, and consolidation decisions:
episode/task id, robot pose, the current timestamp, and a free-form ``extra``
bag for anything else (e.g. current question, last action, VLM confidence).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..typing import Metadata


class MemoryContext:
    """Immutable-by-convention environment snapshot.

    Fields are not enforced immutable (so policies can annotate ``extra`` as
    they flow through); callers should treat a context as read-only.
    """

    __slots__ = ("episode_id", "task_id", "pose", "timestamp", "step", "extra")

    def __init__(
        self,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        pose: Optional[tuple] = None,
        timestamp: Optional[float] = None,
        step: int = 0,
        extra: Optional[Metadata] = None,
    ) -> None:
        self.episode_id = episode_id
        self.task_id = task_id
        self.pose = tuple(pose) if pose is not None else None
        self.timestamp = float(timestamp) if timestamp is not None else None
        self.step = int(step)
        self.extra: Metadata = dict(extra) if extra else {}

    # -- dunder ------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MemoryContext(episode={self.episode_id!r}, task={self.task_id!r}, "
            f"t={self.timestamp}, step={self.step})"
        )

    # -- convenience ------------------------------------------------------- #
    def child(self, **overrides: Any) -> "MemoryContext":
        """Return a shallow copy with selected fields overridden.

        Handy for consolidation passes that want to spawn a new context per
        edge without mutating the original.
        """
        kwargs: Dict[str, Any] = dict(
            episode_id=self.episode_id,
            task_id=self.task_id,
            pose=self.pose,
            timestamp=self.timestamp,
            step=self.step,
            extra=dict(self.extra),
        )
        kwargs.update(overrides)
        return MemoryContext(**kwargs)


__all__ = ["MemoryContext"]
