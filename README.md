# 通用EQA记忆框架与VLA智能体记忆修正

> **目标**：让 VLA/EQA 系统主动识别记忆中的错误并修复。
> **路径**：通过统一记忆框架 `unimem` 把不同方法的记忆模块抽象整合，使跨层 / 跨方法的错误记忆检索与修正成为可能。
> **状态**：框架骨架已实现，且已复现三种已有EQA/VLA方法；等待进行实机测试

---

## 项目结构

```
unified_memory_framework/
├── research/          # 文献调研存档（~55 篇 EQA/具身Agent 论文）
│   ├── README.md      # 存档索引 + 核实修正记录
│   ├── research_notes.md  # 研究结论
│   ├── papers_matrix.md   # 论文 × 记忆模块分类矩阵
│   └── references.bib     # 格式化引用
│
├── unimem/            # 通用记忆框架
│   ├── core/          # 数据类型与抽象
│   ├── graph/         # 记忆图实现
│   ├── policies/      # 读写策略与参考实现
│   ├── factory/       # 工厂方法
│   ├── reference/     # 唯一参考实现 ListEpisodicMemory
│   └── tests/         # 测试
│
├── reproductions/     # 三种已有方法实现与重构
    ├── _common/       # Mock
    ├── r4/            # R^4 (arXiv 2512.15940) — 4D知识库
    ├── clivis/        # CLiViS (CVPR 2026) — 三种模块记忆+迭代修正
    ├── videohv/       # VideoHV-Agent (CVPR 2026) — 图像注释->隐式记忆
    └── tests/         # 对比测试
```

---

## 研究背景

- **问题**：VLA/EQA 智能体运行中累积记忆错误——幻觉对象、位置漂移、信息过时、关系错误、语义误分类、合并/沉淀引入的假事实——导致问答与导航精度下降。
- **差距**：现有方法只有**预防性**机制（写入门控、单次 VLM 验证），无系统性**纠正性**机制；各方法记忆彼此孤立、表征各异，无法跨方法纠错。
- **方法**：实现了统一的多模块记忆框架（WM / SG / GM / EM / SM / PM），支持：
  1. **多模块一致性校验**——处理不同模块记忆交互（如 SG 的"杯子在桌上" vs GM 的桌面几何）
  2. **即插即用的记忆修正模块**——通过统一接口适配不同 VLA/EQA 架构
  3. **跨方法记忆对比**——集成式错误检测（场景图法 vs 地图法 对同一环境的记忆差异）

---

## `unimem/` — 通用记忆框架

纯Python标准库实现，无第三方依赖。详细文档见 [unimem/README.md](unimem/README.md)（设计原则、关键决策、API 速查）。

### 核心设计

- **6种记忆槽位**（`MemorySlot`）：`WM` / `SG` / `GM` / `EM` / `SM` / `PM`
- **模块即图节点**：每个 `MemoryModule` 实例是图中一个节点；数据流/沉淀/索引是边
- **5种关系边**（`EdgeKind`）：`FEEDS`（数据流）/ `CONSOLIDATES_TO`（沉淀）/ `INDEXES` / `REFERENCES` / `SUBSUMES`（层级）
- **3个核心图算法**（`MemoryGraph`）：
  1. **扇出-读** `read(query)` — 广播 query 到所有匹配 slot_filter 的节点，结果带溯源
  2. **扇入-写** `write(entry, source)` — BFS 沿 FEEDS 边传播，VISITED 防环，三级写策略门控（边>模块>图默认）
  3. **沉淀遍历** `run_consolidation_pass(ctx)` — 沿 CONSOLIDATES_TO 提取并写入，后置遗忘扫描

### 关键设计决策

1. 自定义图而非 networkx：节点少（3-8 个）+ 边带类型与策略，邻接表足够
2. 槽位 ABC 精简：基类 4 抽象方法（`write/read/clear/stats`），每个槽位 ABC 仅加 2-3 个
3. 沉淀策略挂在边而非模块：同一 EM 可向 SM 抽事实、向 GM 抽空间模式
4. FEEDS 默认恒等传播：边策略只门控不变换
5. `MultiAxisIndex` 是工具非强制：场景图等模块可有自己的树结构

---

## `reproductions/` — 真实方法复现

把三种风格各异的 EQA/VideoQA 方法的**记忆系统**复现为 `unimem.MemoryModule` 子类。所有测试可用 mock 跑通，**无需 GPU、模型权重、API key**。详细文档见 [reproductions/README.md](reproductions/README.md)。

| 方法 | 原始形态 | 复现产出 | 框架验证点 |
|------|---------|---------|----------|
| **R4** (arXiv 2512.15940) | 仅论文 | `R4KnowledgeDatabase`（4D 知识库）+ storage/retrieval pipeline | 复现论文 Eq.5 dedup（ε_c + δ_s 双阈值，纯 stdlib 向量运算） |
| **CLiViS** (CVPR 2026) | 完整源码 + Neo4j | `TimeWorkingMemory` + `NavigationGraph` + `RelationGraph`（纯 Python 属性图替代 Neo4j） | 三模块分离 + RelationGraph 双 ABC 身份（SG+SM） |
| **VideoHV-Agent** (CVPR 2026) | 完整源码但无记忆模块 | `VideoSummaryMemory`（clip 注释 = 情景记忆）+ `VerificationTraceMemory`（验证轨迹 = 短期 EM） | 把 stateless pipeline 重构为隐式记忆系统 |

### 复现证明了什么

- 框架 plug-in 接口足够通用——三种风格迥异的方法都能在不改 unimem 源码的情况下接入
- slot 抽象粒度合适——既不强制每个方法 6 槽位全占，也避免把异构记忆硬塞进一个槽
- 图边带策略的设计足够灵活——从 R4 的单 DB 节点到 CLiViS 的三节点 fan-out 都能表达

---

## 快速开始

### 运行测试

```bash
cd /home/eg4/pwttt/cdx

# 仅 unimem 框架（149 测试）
python -m unittest discover -s unimem/tests -v

# 仅 reproductions（177 测试）
python -m unittest discover -s reproductions -v

# 全部（326 测试，~0.02 秒）
python -m unittest discover -s unimem/tests -t . && \
python -m unittest discover -s reproductions -t .
```

### 最小示例：构建一个 unimem 图

```python
from unimem import (
    MemorySlot, MemoryEntry, MemoryContext, QueryBuilder,
    MemoryGraph, MemoryNode, MemoryEdge, EdgeKind,
    ListEpisodicMemory,
)

g = MemoryGraph()
g.add_node(MemoryNode("wm", MemorySlot.WM, _MyWorkingMemory()))
g.add_node(MemoryNode("em", MemorySlot.EM, ListEpisodicMemory(timescales=(60.0,))))
g.add_edge(MemoryEdge("wm", "em", EdgeKind.FEEDS))

# 写入观察 — 自动沿 FEEDS 传播到 EM
g.write(
    MemoryEntry("obs1", "saw a red chair",
                semantic_keys=["chair", "red"],
                spatial_keys=[(1.5, 2.5)],
                temporal_keys=[1.0]),
    MemoryContext(timestamp=1.0),
    source_node_id="wm",
)
```

### 接入真实模型

把 `MockLLM` 替换为任何 `(prompt: str, **kw) -> str` 的可调用对象即可：

```python
from my_real_client import real_llm, real_vlm
from reproductions.r4.pipeline import R4Pipeline
from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase

db = R4KnowledgeDatabase(vlm_describe=real_vlm)
pipe = R4Pipeline(db=db, vlm=real_vlm)
```

CLiViS 的 LLM/VLM、R4 的 embedding、VideoHV 的 vision-tools 同理可替换。

---

## 关键参照方法

研究主线的对照基准——每种方法的现有预防性机制，及其作为修正研究切入点的局限：

| 方法 | 现有机制 | 局限（修正研究的切入点） |
|------|---------|------------------------|
| INHerit-SG | 检索时的 VLM 视觉验证（+8.6%） | 仅在检索时验证候选，非系统性纠错 |
| MemoryEQA | 写入三条件门控 + 熵停止 | 只防新错误进入，不修已有错误 |
| WorldMM | 情景→语义 consolidation（去冲突） | 只在抽象层去重，不回查底层感知 |
| Pred-EQA | 预测-修正闭环（修剪错误预测） | 修正的是规划预测，不是记忆内容 |
| DovSG | 局部子图更新（动态环境） | 只处理物体移动，不处理语义/关系错误 |
| DynaMem | 显式变化检测 | 检测变化但不纠幻觉/错分类 |

---

## 当前研究阶段

- ✅ 文献调研（~55 篇，2025-2026 多记忆融合为主）
- ✅ 通用记忆框架 v2 设计（6 槽位 + 3 横切机制）
- ✅ 框架骨架实现：`unimem/` 包（纯 stdlib，149 测试全过）
- ✅ 三种真实方法复现：`reproductions/`（177 测试，零第三方依赖）
- 🔄 当前：将框架映射到记忆修正应用；分析每层在具体方法中的功能
- ⏳ 下一步：定义修正模块接口；选择实验测试床（OpenEQA / HM3D / Habitat）；形式化错误类型学

---

## 工作约定

- **语言**：中文为主，技术术语保留英文
- **方法学**：文献驱动；任何框架/机制设计需有论文支撑（可在 `papers_matrix.md` 追溯）
- **范围**：以 EQA 为主要测试域（有明确 QA 正确性信号便于评估修正效果），向 VLA 通用化
- **代码**：避免过度工程化；不为假设性未来需求设计；不添加未被请求的 docstring/类型注解/特征
- **依赖**：`unimem/` 纯 stdlib 硬约束；`reproductions/` 同样零硬依赖（neo4j/numpy/torch 均用 stdlib 替代）
- **Python**：3.9+ 兼容（`from __future__ import annotations`，无 `X | Y` 运行时注解）

更多详细工作指示见 [CLAUDE.md](CLAUDE.md)。
