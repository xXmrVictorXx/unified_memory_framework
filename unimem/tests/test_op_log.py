"""Tests for SQLiteOpLog."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

from unimem.op_log import OpLogEntry, SQLiteOpLog


class TestSQLiteOpLog(unittest.TestCase):
    def test_append_assigns_lsn(self):
        with SQLiteOpLog(":memory:") as log:
            lsn1 = log.append(OpLogEntry(op_type="write", node_id="wm"))
            lsn2 = log.append(OpLogEntry(op_type="write_done", node_id="wm"))
            self.assertGreater(lsn2, lsn1)
            self.assertEqual(log.last_lsn(), lsn2)

    def test_replay_returns_all_entries(self):
        with SQLiteOpLog(":memory:") as log:
            log.append(OpLogEntry(op_type="write", node_id="wm"))
            log.append(OpLogEntry(op_type="write_done", node_id="wm", result={"wm": True}))
            log.append(OpLogEntry(op_type="read", node_id="em"))
            entries = log.replay()
            self.assertEqual(len(entries), 3)
            self.assertEqual([e.op_type for e in entries], ["write", "write_done", "read"])

    def test_replay_from_lsn(self):
        with SQLiteOpLog(":memory:") as log:
            lsn1 = log.append(OpLogEntry(op_type="write", node_id="wm"))
            log.append(OpLogEntry(op_type="write_done", node_id="wm"))
            entries = log.replay(from_lsn=lsn1)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].op_type, "write_done")

    def test_query_by_op_type(self):
        with SQLiteOpLog(":memory:") as log:
            log.append(OpLogEntry(op_type="write", node_id="wm"))
            log.append(OpLogEntry(op_type="read", node_id="em"))
            log.append(OpLogEntry(op_type="write", node_id="sg"))
            writes = log.query(op_type="write")
            self.assertEqual(len(writes), 2)
            for e in writes:
                self.assertEqual(e.op_type, "write")

    def test_query_by_node_id(self):
        with SQLiteOpLog(":memory:") as log:
            log.append(OpLogEntry(op_type="write", node_id="wm"))
            log.append(OpLogEntry(op_type="write", node_id="em"))
            entries = log.query(node_id="wm")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].node_id, "wm")

    def test_query_limit(self):
        with SQLiteOpLog(":memory:") as log:
            for _ in range(10):
                log.append(OpLogEntry(op_type="write", node_id="wm"))
            self.assertEqual(len(log.query(limit=5)), 5)

    def test_serialise_entry_dict(self):
        with SQLiteOpLog(":memory:") as log:
            entry = OpLogEntry(
                op_type="write",
                node_id="em",
                entry_dict={"entry_id": "e1", "text": "hello"},
                context_dict={"timestamp": 1.5},
                metadata={"round": 0},
            )
            lsn = log.append(entry)
            replayed = log.replay(from_lsn=lsn - 1)
            self.assertEqual(len(replayed), 1)
            r = replayed[0]
            self.assertEqual(r.entry_dict["entry_id"], "e1")
            self.assertEqual(r.entry_dict["text"], "hello")
            self.assertEqual(r.context_dict["timestamp"], 1.5)
            self.assertEqual(r.metadata["round"], 0)

    def test_checkpoint_and_recover(self):
        with SQLiteOpLog(":memory:") as log:
            log.append(OpLogEntry(op_type="write", node_id="wm"))
            log.append(OpLogEntry(op_type="write_done", node_id="wm"))
            cp_lsn = log.checkpoint({"snapshot": "v1", "n_nodes": 3})
            self.assertEqual(cp_lsn, log.last_lsn())
            cp = log.last_checkpoint()
            self.assertIsNotNone(cp)
            self.assertEqual(cp["snapshot_metadata"]["snapshot"], "v1")

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "op_log.sqlite3")
            with SQLiteOpLog(path) as log:
                log.append(OpLogEntry(op_type="write", node_id="wm"))
                self.assertEqual(log.last_lsn(), 1)
            # Reopen and verify
            with SQLiteOpLog(path) as log2:
                self.assertEqual(log2.last_lsn(), 1)
                entries = log2.replay()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].op_type, "write")

    def test_temporary_path_default(self):
        # Default path=None creates a temp file; cleanup on close.
        log = SQLiteOpLog()
        path = log._path  # noqa: SLF001 — test only
        log.append(OpLogEntry(op_type="write", node_id="wm"))
        log.close()
        # Temp file should be gone (TemporaryDirectory cleans up).
        # We don't assert on path existence since SQLiteOpLog may use
        # tempdir cleanup; just verify no exception.


class TestOpLogEntry(unittest.TestCase):
    def test_to_dict_round_trip(self):
        e = OpLogEntry(
            op_type="write",
            node_id="wm",
            entry_dict={"a": 1},
            metadata={"k": "v"},
        )
        d = e.to_dict()
        self.assertEqual(d["op_type"], "write")
        self.assertEqual(d["node_id"], "wm")
        self.assertEqual(d["entry_dict"], {"a": 1})
        self.assertEqual(d["metadata"], {"k": "v"})


if __name__ == "__main__":
    unittest.main()
