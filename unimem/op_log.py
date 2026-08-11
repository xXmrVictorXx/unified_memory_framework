"""``unimem.op_log`` — operation log for cross-backend atomicity + audit.

The OpLog is the WAL (write-ahead log) that :class:`~unimem.graph.graph.MemoryGraph`
appends to **before** dispatching mutations to its
:class:`~unimem.graph_storage.base.GraphStorage`. On crash recovery, the
graph replays the log and re-issues every operation; because backends use
MERGE / upsert semantics, replay is idempotent.

Design points:

* ``OpLogEntry`` is a tiny ``__slots__`` record with a monotonic ``lsn``
  (log sequence number) and a wall-clock ``timestamp``.
* :class:`SQLiteOpLog` is the only concrete implementation; sqlite3 is in
  the stdlib and good enough for our throughput needs (EQA workloads are
  not write-heavy).
* Subscribers can ``replay()`` the whole log or ``query()`` with filters
  for audit / debugging.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# OpLogEntry
# --------------------------------------------------------------------------- #
class OpLogEntry:
    """A single logged operation.

    Fields are intentionally permissive — most are optional depending on
    ``op_type``. For example, a ``"write"`` carries ``entry_dict`` +
    ``context_dict`` + ``node_id``; a ``"write_done"`` carries ``result``.
    """

    __slots__ = (
        "lsn",
        "timestamp",
        "op_type",
        "node_id",
        "entry_dict",
        "context_dict",
        "result",
        "metadata",
    )

    def __init__(
        self,
        op_type: str,
        node_id: Optional[str] = None,
        entry_dict: Optional[Dict[str, Any]] = None,
        context_dict: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        lsn: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.op_type = str(op_type)
        self.node_id = node_id
        self.entry_dict = dict(entry_dict) if entry_dict else None
        self.context_dict = dict(context_dict) if context_dict else None
        self.result = result
        self.metadata = dict(metadata) if metadata else None
        self.lsn = lsn  # populated by OpLog.append
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lsn": self.lsn,
            "timestamp": self.timestamp,
            "op_type": self.op_type,
            "node_id": self.node_id,
            "entry_dict": self.entry_dict,
            "context_dict": self.context_dict,
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "OpLogEntry":
        """Reconstruct from a sqlite row (lsn, timestamp, op_type, node_id,
        entry_dict, context_dict, result, metadata).
        """
        def _loads(x):
            if x is None or x == "":
                return None
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return None

        return cls(
            op_type=row["op_type"],
            node_id=row["node_id"],
            entry_dict=_loads(row["entry_dict"]),
            context_dict=_loads(row["context_dict"]),
            result=_loads(row["result"]),
            metadata=_loads(row["metadata"]),
            lsn=row["lsn"],
            timestamp=row["timestamp"],
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OpLogEntry(lsn={self.lsn}, op_type={self.op_type!r}, "
            f"node_id={self.node_id!r})"
        )


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #
class OpLog(ABC):
    """Abstract operation log."""

    @abstractmethod
    def append(self, entry: OpLogEntry) -> int:
        """Persist ``entry`` and return its assigned LSN."""

    @abstractmethod
    def replay(self, from_lsn: int = 0) -> List[OpLogEntry]:
        """Return every entry with ``lsn > from_lsn``, in order."""

    @abstractmethod
    def query(
        self,
        op_type: Optional[str] = None,
        node_id: Optional[str] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        limit: int = 100,
    ) -> List[OpLogEntry]:
        """Filtered query for audit / inspection."""

    @abstractmethod
    def checkpoint(self, snapshot_metadata: Dict[str, Any]) -> int:
        """Record a checkpoint; returns the LSN at which it was taken."""

    @abstractmethod
    def last_lsn(self) -> int:
        """Highest LSN persisted (0 if empty)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


# --------------------------------------------------------------------------- #
# SQLiteOpLog
# --------------------------------------------------------------------------- #
class SQLiteOpLog(OpLog):
    """Persistent OpLog backed by a single sqlite3 file.

    Tables:

    * ``op_log(lsn PK AUTOINCREMENT, timestamp, op_type, node_id,
              entry_dict, context_dict, result, metadata)``
    * ``checkpoints(lsn PK, timestamp, snapshot_metadata)``

    Indexes on ``op_type``, ``node_id``, ``timestamp``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Open (or create) an op log at ``path``.

        If ``path`` is None, a temporary file is used and will be removed on
        :meth:`close`. If ``path`` is ``":memory:"``, sqlite runs entirely in
        RAM (useful for tests).
        """
        self._owns_path = path is None
        if path is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="unimem-oplog-")
            path = str(Path(self._tmpdir.name) / "op_log.sqlite3")
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS op_log (
                lsn INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                op_type TEXT NOT NULL,
                node_id TEXT,
                entry_dict TEXT,
                context_dict TEXT,
                result TEXT,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_op_log_op_type ON op_log(op_type);
            CREATE INDEX IF NOT EXISTS ix_op_log_node_id ON op_log(node_id);
            CREATE INDEX IF NOT EXISTS ix_op_log_timestamp ON op_log(timestamp);

            CREATE TABLE IF NOT EXISTS checkpoints (
                lsn INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                snapshot_metadata TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def append(self, entry: OpLogEntry) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO op_log (timestamp, op_type, node_id, entry_dict, "
            "context_dict, result, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp if entry.timestamp is not None else time.time(),
                entry.op_type,
                entry.node_id,
                _dumps(entry.entry_dict),
                _dumps(entry.context_dict),
                _dumps(entry.result),
                _dumps(entry.metadata),
            ),
        )
        self._conn.commit()
        lsn = cur.lastrowid or self.last_lsn()
        entry.lsn = lsn
        return lsn

    def replay(self, from_lsn: int = 0) -> List[OpLogEntry]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM op_log WHERE lsn > ? ORDER BY lsn ASC",
            (int(from_lsn),),
        )
        return [OpLogEntry.from_row(row) for row in cur.fetchall()]

    def query(
        self,
        op_type: Optional[str] = None,
        node_id: Optional[str] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        limit: int = 100,
    ) -> List[OpLogEntry]:
        clauses = []
        params: List[Any] = []
        if op_type is not None:
            clauses.append("op_type = ?")
            params.append(op_type)
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if t_min is not None:
            clauses.append("timestamp >= ?")
            params.append(float(t_min))
        if t_max is not None:
            clauses.append("timestamp <= ?")
            params.append(float(t_max))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT * FROM op_log{where} ORDER BY lsn DESC LIMIT ?",
            params,
        )
        return [OpLogEntry.from_row(row) for row in cur.fetchall()]

    def checkpoint(self, snapshot_metadata: Dict[str, Any]) -> int:
        lsn = self.last_lsn()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO checkpoints (lsn, timestamp, snapshot_metadata) "
            "VALUES (?, ?, ?)",
            (lsn, time.time(), json.dumps(snapshot_metadata, default=str)),
        )
        self._conn.commit()
        return lsn

    def last_checkpoint(self) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT lsn, timestamp, snapshot_metadata FROM checkpoints "
            "ORDER BY lsn DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "lsn": row["lsn"],
            "timestamp": row["timestamp"],
            "snapshot_metadata": json.loads(row["snapshot_metadata"]),
        }

    def last_lsn(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT MAX(lsn) AS m FROM op_log")
        row = cur.fetchone()
        return int(row["m"]) if row["m"] is not None else 0

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
        if self._owns_path and hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()

    # Allow use as a context manager
    def __enter__(self) -> "SQLiteOpLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _dumps(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        return json.dumps(x, default=str)
    except (TypeError, ValueError):
        return None


__all__ = ["OpLog", "OpLogEntry", "SQLiteOpLog"]
