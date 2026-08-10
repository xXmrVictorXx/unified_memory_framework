# unimem — 通用 EQA 记忆框架

纯 Python 标准库实现的通用具身问答（EQA）记忆框架。**零第三方依赖，Python 3.9+ 兼容，149 测试全过**。

## 设计哲学

> 不同 EQA 方法的记忆模块"难以向下兼容，但可以通过抽象的方法进行向上兼容"——
> 即，通用记忆框架为众多不同类型的记忆**预设位置（槽位）**，不同 EQA 方法按需存储记忆。

框架定义**接口契约**而非具体实现。不同方法 plug-in 自己的实现即向上兼容。

## 包结构

```
unimem/
├── core/          # 数据类型 + ABC
│   ├── slots.py               # MemorySlot 枚举（6 槽位）
│   ├── entry.py               # MemoryEntry + MultiAxisIndex（三轴可索引，R4 启发）
│   ├── context.py             # MemoryContext（episode/task/pose/timestamp）
│   ├── query.py               # Query / QueryBuilder / QueryResult
│   ├── module.py              # MemoryModule ABC（核心契约）
│   └── slot_abc.py            # 6 个槽位专用 ABC（每个仅加 2-3 个方法）
├── graph/         # 图组织
│   ├── edge.py                # EdgeKind 枚举 + MemoryEdge
│   ├── node.py                # MemoryNode
│   ├── graph.py               # MemoryGraph（核心：三个图算法）
│   └── builder.py             # GraphSpec/NodeSpec/EdgeSpec + MemoryGraphBuilder
├── policies/      # 横切策略
│   ├── write_policy.py        # WritePolicy + AlwaysWrite/NeverWrite/LambdaWritePolicy
│   ├── read_policy.py         # ReadPolicy + ConcatRead/FirstNonEmptyRead
│   ├── consolidation_policy.py# ConsolidationPolicy + Passthrough
│   └── forget_policy.py       # ForgetPolicy + NoOp
├── factory/       # 工厂与注册表
│   ├── registry.py            # Registry（按 (slot, impl_name) 注册）
│   └── memory_factory.py      # MemoryFactory 门面
├── reference/     # 参考实现
│   ├── episodic_memory.py     # ListEpisodicMemory（多时间尺度+多轴索引）
│   ├── forget_fifo.py         # FIFOForgetPolicy
│   └── consolidate_extract.py # ExtractFactsConsolidationPolicy（EM→SM 事实抽取）
└── tests/         # 9 个测试文件，149 测试
```

## 核心抽象

### 6 个记忆槽位

```
WM = working_memory       # 当前观测、任务状态、近期上下文
SG = scene_graph          # 层级对象-关系拓扑
GM = spatial_geometric    # 度量地图、占用、导航可行
EM = episodic             # 时序事件、观察序列
SM = semantic             # 事实、规则、常识
PM = procedural           # 动作策略、能力 profile
```

来源：~55 篇 EQA/具身Agent 论文调研（见 `research/research_notes.md`）。最接近占满 6 槽位的是 HIMM(5) 和 CRAM 2.0(5)——通用框架的"通用性"价值真实存在。

### 5 种边

```
FEEDS            # 数据流：WM → EM
CONSOLIDATES_TO  # 沉淀：EM → SM（携带 ConsolidationPolicy）
INDEXES          # 索引：SG 节点 → GM 区域
REFERENCES       # 跨模块指针（任意）
SUBSUMES         # 层级包含：room → objects
```

### 三个核心图算法

1. **扇出读** `MemoryGraph.read(query) -> List[QueryResult]`
   - 按 `query.slot_filter` 选目标节点（无则全部节点）
   - 每节点调 `module.read(query)`，结果带 `source_node_id` / `source_slot` 溯源
2. **扇入写** `MemoryGraph.write(entry, context, source_node_id=None) -> Dict[str, bool]`
   - BFS 沿 FEEDS 边传播；VISITED 集合防环；三级写策略门控（边>模块>图默认）
   - 默认传播同一 entry 对象（不变换）；如需变换由目标模块 `write()` 内部处理
3. **沉淀遍历** `MemoryGraph.run_consolidation_pass(context) -> Dict[str, int]`
   - 沿 CONSOLIDATES_TO 边：边携带 ConsolidationPolicy 则用它，否则回退到源模块 `consolidate()`
   - 遍历后对所有节点跑 forget_policy（容量管理/VoI 淘汰）

## API 速查

```python
from unimem import (
    MemorySlot, MemoryEntry, MemoryContext, QueryBuilder, Query,
    MemoryGraph, MemoryNode, MemoryEdge, EdgeKind,
    Registry, MemoryGraphBuilder, GraphSpec, NodeSpec, EdgeSpec,
    ListEpisodicMemory, FIFOForgetPolicy, ExtractFactsConsolidationPolicy,
)

# 构建图（命令式）
g = MemoryGraph()
g.add_node(MemoryNode("wm", MemorySlot.WM, _MyWM()))
g.add_node(MemoryNode("em", MemorySlot.EM, ListEpisodicMemory(timescales=(60.0, 600.0))))
g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))

# 或声明式（从 Registry 实例化）
registry = Registry()
registry.register_module(MemorySlot.EM, "list", ListEpisodicMemory)
g = MemoryGraphBuilder(registry).build(GraphSpec(
    nodes=[NodeSpec("wm", MemorySlot.WM, "stub"),
           NodeSpec("em", MemorySlot.EM, "list", kwargs={"timescales": (60.0,)})],
    edges=[EdgeSpec("wm", "em", EdgeKind.FEEDS)],
))

# 写（扇入，自动沿 FEEDS 传播）
g.write(MemoryEntry("obs1", "saw a red chair",
                    semantic_keys=["chair", "red"],
                    spatial_keys=[(1.5, 2.5)],
                    temporal_keys=[1.0]),
        MemoryContext(timestamp=1.0),
        source_node_id="wm")

# 读（扇出，按 slot_filter 分发）
results = g.read(QueryBuilder().with_slot(MemorySlot.EM).with_semantic("chair").build())

# 沉淀（沿 CONSOLIDATES_TO）
result = g.run_consolidation_pass(MemoryContext())
```

## 关键设计决策

1. **自定义图而非 networkx**：零依赖硬约束 + 记忆图节点少（3-8 个）+ 边带类型与策略，自定义邻接表 `Dict[str, List[MemoryEdge]]` 足够。
2. **槽位 ABC 精简**：基类 4 抽象方法（`write` / `read` / `clear` / `stats`），每个槽位 ABC 仅加 2-3 个——避免实现单槽位还要实现 50+ 方法的负担。
3. **沉淀策略挂在边而非模块**：同一 EM 可向不同目标做不同沉淀（向 SM 抽事实、向 GM 抽空间模式）。
4. **FEEDS 默认恒等传播**：边策略只门控不变换；变换由目标模块 `write()` 内部处理，保证数据流透明。
5. **MultiAxisIndex 是工具非强制**：场景图等模块可有自己的树结构，不应被强制用倒排索引。支持 `within_axis_op="intersect"|"union"`（默认 intersect，匹配 EQA 多词查询的 AND 语义）。
6. **单一 Registry 按 (slot, impl_name)**：比 6 个分槽注册表更简单，支持 `list_implementations(slot)` 查询。
7. **GraphSpec dataclass + from_dict**：dataclass 便于 IDE/类型检查，dict 便于实验驱动架构搜索（JSON/YAML 可自行加载）。
8. **Python 3.9 兼容**：所有文件 `from __future__ import annotations`；用 `typing.Optional/List/Dict/Tuple`，不用 `X | Y` 或运行时 `list[str]`。

## 测试

```bash
python -m unittest discover -s unimem/tests -v
# Ran 149 tests in 0.007s — OK
```

测试覆盖：

| 文件 | 内容 |
|------|------|
| `test_entry.py` | 多轴索引（intersect/union）+ remove + clear |
| `test_module.py` | ABC 合规 + 最小具体模块 |
| `test_graph.py` | **三个核心算法**（最重要）：拓扑、树查询、扇出读、扇入写（含 cycle/edge-policy/event-trigger）、沉淀遍历 |
| `test_policies.py` | 4 类策略 ABC + 默认实现 |
| `test_factory.py` | Registry 注册 / decorator / list_implementations |
| `test_builder.py` | 声明式 GraphSpec 构建（dataclass + dict round-trip） |
| `test_reference.py` | ListEpisodicMemory 端到端（多时间尺度 + 多轴 + FIFO forget + ExtractFacts 沉淀） |
| `test_plug_in.py` | 模拟 EQA 方法 plug-in 场景（4 节点图端到端） |

## 接入第三方方法

参见 `reproductions/` ——三种真实 EQA 方法（R4 / CLiViS / VideoHV-Agent）的完整接入示例。
