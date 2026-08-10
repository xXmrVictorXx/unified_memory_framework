# Reproductions — 将真实 EQA 方法接入 unimem

本目录把三种风格各异的 EQA/VideoQA 方法的**记忆系统**复现为 `unimem.MemoryModule`
子类，并提供带可注入 LLM/VLM 调用子的 pipeline 骨架。**所有测试可用 mock 跑通，
无需 GPU、模型权重、API key**——总计 177 个测试全部通过，零第三方依赖（除 unimem 本身）。

## 目录结构

```
reproductions/
├── _common/              # MockLLM / MockVLM / MockEmbedding / MockVisionTools
├── clivis/               # CLiViS (CVPR 2026) — 完整源码 → 3 个记忆模块
├── r4/                   # R4 (arXiv 2512.15940) — 论文 → 4D 知识库
├── videohv/              # VideoHV-Agent (CVPR 2026) — 重构为隐式记忆系统
└── tests/                # 跨方法对比集成测试
```

## 三种方法 × 三种记忆风格

| 方法 | 原始代码 | 记忆风格 | unimem 槽位映射 | 测试数 |
|------|---------|---------|----------------|-------|
| **R4** | 仅论文 | 单一 4D 知识库（SEM/SPA/TEM 三轴索引）| GM (含 EM+SM 数据) | 45 |
| **CLiViS** | 完整源码（含 Neo4j） | 显式三模块：时间工作记忆 / 导航图 / Neo4j 关系图 | WM + SG + GM + SM | 79 |
| **VideoHV-Agent** | 完整源码但无记忆模块 | 重构：预计算视频注释 = 情景记忆；验证轨迹 = 短期 EM | EM + EM | 36 |
| **跨方法集成** | — | — | — | 17 |

## 三种方法学到的框架验证

| 验证点 | R4 | CLiViS | VideoHV |
|--------|----|--------|---------|
| 单一 MemoryModule 子类化 | ✓ R4KnowledgeDatabase | ✓ TimeWorkingMemory / NavigationGraph / RelationGraph | ✓ VideoSummaryMemory / VerificationTraceMemory |
| 同时实现 2+ 个 slot ABC | ✓ SpatialGeometricABC + 隐式多轴 | ✓ WM+Episodic / SG+Semantic / SG+Spatial | ✓ Episodic (两份) |
| 通过 unimem MemoryGraph 编排 | ✓ (WM→DB→SM, FEEDS+CONSOLIDATES_TO) | ✓ (WM→SG, WM→GM, SG→GM, FEEDS+INDEXES) | ✓ (trace→summary, REFERENCES) |
| 扇入写沿 FEEDS 自动传播 | ✓ | ✓ | N/A (无 FEEDS 边) |
| 扇出读按 slot_filter 分发 | ✓ | ✓ | ✓ |
| 沉淀 (CONSOLIDATES_TO) | ✓ (DB→SM) | ✓ (pipeline 内部） | N/A |
| Pipeline 端到端跑通（mock） | ✓ storage + retrieval loop | ✓ init + iterative refinement | ✓ hypothesis-verification loop |
| LLM/VLM 可注入 | ✓ VLMFn + DecompFn | ✓ LLMFn + VLMFn | ✓ LLMFn + VisionTools |
| 零第三方硬依赖 | ✓ numpy-free (纯 stdlib 向量运算) | ✓ Neo4j-free (纯 Python 属性图) | ✓ |

## 关键设计决策

### R4
- **ObjectRecord dataclass** 三轴分离：SEM (description + embedding), SPA (centroid + extent), TEM (timestamps)
- **复现论文 Eq. 5 dedup**：空间近邻 (ε_c) + 语义余弦相似度 (δ_s) 双阈值，纯 stdlib 向量运算
- **SimpleSLAMMap**：MapAnything 的接口级 stub——只保留 R4 需要的 special points / nearest-neighbors / directional filter
- **可注入 embedding**：默认 MockEmbedding，生产环境通过 `set_default_embedding()` 切换 sentence-transformers / OpenAI

### CLiViS
- **三个 MemoryModule 子类** 而非一个——CLiViS 原本就是三模块分离设计，不应强行合并
- **Neo4j → 纯 Python 属性图**：完整保留 Person/Object/Area/Activity 四类节点 + 全部关系类型 + 动作链表
- **NavigationGraph 的时空重叠匹配**：area 通过 time_range overlap 自动归入对应 period
- **每个模块双重 ABC 身份**：RelationGraph = SceneGraphABC + SemanticABC；TimeWorkingMemory = WorkingABC + EpisodicABC

### VideoHV-Agent
- **核心创新**：把 stateless pipeline 重新框定为隐式记忆系统
- **VideoSummaryMemory = EM**：clip 注释直接映射到三轴 MemoryEntry (语义=对象标签，时间=clip 边界)
- **VerificationTraceMemory = 短期 EM**：原本的局部变量 `verification_trace_text` / `prior_hypothesis_lines` 变为可查询的 episodic store
- **答用户原始问题**："VideoHV-Agent 把输入视频切片并生成注释，这何尝不是一种记忆？"——是，框架完全支持

## 设计原则

1. **零硬依赖**：所有真实依赖（torch / transformers / neo4j / openai）均为可选 import。
   - R4 用 stdlib 替代 numpy（向量运算 + 余弦相似度）
   - CLiViS 用纯 Python 属性图替代 Neo4j
   - VideoHV 用纯 stdlib
   - 所有 LLM/VLM 调用通过 `_common/mocks.py` 注入
2. **Pipeline 可跑通**：每个方法的 agent loop 都能跑完整流程（用 mock），产出结构化结果
3. **忠实于原方法**：数据结构、写入/读取语义、关键 prompt 都尽量贴原实现；只在必要处做 unimem 接口适配
4. **可观测**：所有记忆操作可被 unimem 图的 `summary()` 和 `stats()` 反馈出来

## 运行测试

```bash
cd /home/eg4/pwttt/cdx

# 仅 unimem 框架（149 测试）
python -m unittest discover -s unimem/tests -v

# 仅 reproductions（177 测试）
python -m unittest discover -s reproductions -v

# 全部（326 测试）
python -m unittest discover -s unimem/tests -t . && \
python -m unittest discover -s reproductions -t .
```

## 真实模型接入点

把 `reproductions/_common/mocks.MockLLM` 替换为任何
``(prompt: str, **kw) -> str`` 的可调用对象即可：

```python
from my_real_client import real_llm, real_vlm
from reproductions.r4.pipeline import R4Pipeline
from reproductions.r4.memory.knowledge_db import R4KnowledgeDatabase

db = R4KnowledgeDatabase(vlm_describe=real_vlm)
pipe = R4Pipeline(db=db, vlm=real_vlm)
```

CLiViS 的 LLM/VLM、R4 的 embedding、VideoHV 的 vision-tools 同理可替换。

## 对下游研究的意义

这套复现证明了：

1. **框架 plug-in 接口足够通用** —— 三种风格迥异的方法都能在不改 unimem 源码的情况下接入
2. **slot 抽象粒度合适** —— 既不强制每个方法 6 个槽位全占，也避免把异构记忆硬塞进一个槽
3. **图边带策略** 的设计足够灵活 —— 从 R4 的单 DB 节点到 CLiViS 的三节点 fan-out 都能表达
4. **接下来可做记忆修正研究** —— 三种方法的记忆现都在 unimem 图中可观测、可对比、可统一检索，为
   CLAUDE.md 中描述的"跨槽位一致性校验"和"跨方法记忆对比"打下了基础
