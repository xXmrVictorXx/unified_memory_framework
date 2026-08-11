"""Tests for the unimem.config system."""
from __future__ import annotations

import unittest

from unimem.config import (
    EdgeSpec,
    ModuleSpec,
    StorageConfig,
    UnimemConfig,
    build_graph,
    build_storage,
)


class TestStorageConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = StorageConfig()
        self.assertEqual(cfg.graph["backend"], "memory")
        self.assertEqual(cfg.vector["backend"], "memory")
        self.assertIsNone(cfg.op_log)

    def test_from_dict(self):
        raw = {
            "storage": {
                "graph": {"backend": "neo4j", "uri": "bolt://localhost"},
                "vector": {"backend": "qdrant", "host": "localhost"},
                "op_log": {"enabled": True, "path": "/tmp/op.sqlite3"},
            }
        }
        cfg = UnimemConfig.from_dict(raw)
        self.assertEqual(cfg.storage.graph["backend"], "neo4j")
        self.assertEqual(cfg.storage.vector["backend"], "qdrant")
        self.assertEqual(cfg.storage.op_log["path"], "/tmp/op.sqlite3")


class TestUnimemConfigFromDict(unittest.TestCase):
    def test_modules_and_edges(self):
        raw = {
            "modules": [
                {"node_id": "wm", "slot": "WM", "impl": "default"},
                {"node_id": "em", "slot": "EM"},
            ],
            "edges": [
                {"source": "wm", "target": "em", "kind": "FEEDS"},
            ],
        }
        cfg = UnimemConfig.from_dict(raw)
        self.assertEqual(len(cfg.modules), 2)
        self.assertEqual(cfg.modules[0].node_id, "wm")
        self.assertEqual(cfg.modules[0].slot, "WM")
        self.assertEqual(len(cfg.edges), 1)
        self.assertEqual(cfg.edges[0].kind, "FEEDS")

    def test_empty_dict_yields_empty_config(self):
        cfg = UnimemConfig.from_dict({})
        self.assertEqual(cfg.modules, [])
        self.assertEqual(cfg.edges, [])
        self.assertEqual(cfg.storage.graph["backend"], "memory")


class TestBuildStorage(unittest.TestCase):
    def test_default_builds_in_memory(self):
        gs, vs, op_log = build_storage(StorageConfig())
        from unimem.graph_storage import InMemoryGraphStorage
        from unimem.vector_storage import InMemoryVectorStorage
        self.assertIsInstance(gs, InMemoryGraphStorage)
        self.assertIsInstance(vs, InMemoryVectorStorage)
        self.assertIsNone(op_log)

    def test_op_log_enabled(self):
        cfg = StorageConfig(op_log={"enabled": True, "path": ":memory:"})
        _, _, op_log = build_storage(cfg)
        self.assertIsNotNone(op_log)
        op_log.close()

    def test_op_log_disabled_explicit(self):
        cfg = StorageConfig(op_log={"enabled": False})
        _, _, op_log = build_storage(cfg)
        self.assertIsNone(op_log)


class TestBuildGraph(unittest.TestCase):
    def test_build_graph_with_specs(self):
        from unimem.core.slots import MemorySlot
        from unimem.graph_storage import InMemoryGraphStorage

        gs = InMemoryGraphStorage()
        graph = build_graph(
            graph_storage=gs,
            modules_spec=[
                ModuleSpec(node_id="wm", slot="WM"),
                ModuleSpec(node_id="em", slot="EM"),
            ],
            edges_spec=[EdgeSpec(source="wm", target="em", kind="FEEDS")],
        )
        self.assertIn("wm", graph)
        self.assertIn("em", graph)
        edges = graph.edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, "wm")
        self.assertEqual(edges[0].target_id, "em")

    def test_build_graph_default_storage(self):
        graph = build_graph()
        self.assertEqual(len(graph), 0)


if __name__ == "__main__":
    unittest.main()
