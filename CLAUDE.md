# CLAUDE.md — 项目工作指示

## 项目目标

设计一种 **VLA 智能体记忆修正方法**：让 VLA/EQA 系统主动识别记忆中的错误并修复。
核心技术路径是通过**统一记忆框架**把不同 VLA/EQA 方法的记忆模块抽象整合，
使跨层 / 跨方法的错误记忆检索与修正成为可能。

> 统一记忆框架是**手段**，记忆修正是**目的**。框架的价值最终由"能否支撑修正机制"来衡量。

## 研究主线（问题-差距-方法）

- **问题**：VLA/EQA 智能体运行中累积记忆错误——幻觉对象、位置漂移、信息过时、关系错误、
  语义误分类、合并/沉淀引入的假事实——导致问答与导航精度下降。
- **差距**：现有方法只有**预防性**机制（写入门控、单次 VLM 验证），无系统性**纠正性**机制；
  且各方法记忆彼此孤立、表征各异，无法跨方法纠错。
- **方法**：以统一多槽位记忆框架（WM / SG / GM / EM / SM / PM + 横切机制）为基底，支持：
  1. **跨槽位一致性校验**——一种表征挑战另一种表征（如 SG 的"杯在桌上" vs GM 的桌面几何）
  2. **即插即用记忆修正模块**——通过统一接口适配不同 VLA/EQA 架构
  3. **跨方法记忆对比**——集成式错误检测（场景图法 vs 地图法 对同一环境的记忆差异）

## 必读文件（按顺序）

1. `research/README.md` — 存档索引 + 核实修正记录
2. `research/research_notes.md` — 研究结论（目标、6 槽位框架、7 种融合模式、开放问题）
3. `research/papers_matrix.md` — 论文 × 记忆模块矩阵（写相关工作、找方法参照时查）
4. `research/references.bib` — 51 篇文献，按类别分组，带 `fusion=` 标签可 grep

## 当前研究阶段

- ✅ 文献调研（~55 篇，2025-2026 多记忆融合为主）
- ✅ 通用记忆框架 v2 设计（6 槽位 + 3 横切机制）
- ✅ 框架骨架实现：`unimem/` 包（纯 stdlib，149 测试全过）
- 🔄 当前：将框架映射到记忆修正应用；分析每层在具体方法中的功能
- ⏳ 下一步：定义修正模块接口；选择实验测试床（OpenEQA / HM3D / Habitat）；
  形式化错误类型学

## `unimem/` 框架（已实现）

通用记忆框架的纯 Python 标准库实现。**零第三方依赖**，Python 3.9+ 兼容。

### 核心设计

- **6 槽位**（`MemorySlot` 枚举）：`WM` / `SG` / `GM` / `EM` / `SM` / `PM`
- **模块即节点**：每个 `MemoryModule` 实例是图中一个节点；数据流/沉淀/索引是边
- **5 种边**（`EdgeKind`）：`FEEDS`（数据流）/ `CONSOLIDATES_TO`（沉淀）/ `INDEXES` / `REFERENCES` / `SUBSUMES`（层级）
- **三个核心图算法**（`MemoryGraph`）：
  1. **扇出读** `read(query)` — 广播 query 到所有匹配 slot_filter 的节点，结果带溯源
  2. **扇入写** `write(entry, source)` — BFS 沿 FEEDS 边传播，VISITED 防环，三级写策略门控（边>模块>图默认）
  3. **沉淀遍历** `run_consolidation_pass(ctx)` — 沿 CONSOLIDATES_TO 提取并写入，后置遗忘扫描

### 关键目录

| 路径 | 内容 |
|------|------|
| `unimem/core/` | 数据类型（`MemoryEntry`/`MultiAxisIndex`/`Query`/`MemoryContext`）+ ABC（`MemoryModule` + 6 槽位 ABC） |
| `unimem/graph/` | `MemoryGraph`（核心）+ `MemoryNode`/`MemoryEdge` + 声明式 `GraphSpec`/`MemoryGraphBuilder` |
| `unimem/policies/` | 4 个横切策略 ABC：Write/Read/Consolidation/Forget（每个含默认实现） |
| `unimem/factory/` | `Registry`（按 `(slot, impl_name)` 注册）+ `MemoryFactory` 门面 |
| `unimem/reference/` | 唯一参考实现 `ListEpisodicMemory` + `FIFOForgetPolicy` + `ExtractFactsConsolidationPolicy` |
| `unimem/tests/` | 9 个测试文件，149 测试；`test_plug_in.py` 是端到端 plug-in 场景 |

### 实现约定

- **纯 stdlib 硬约束**：禁止 `import numpy/torch/networkx/scipy/...`。`MultiAxisIndex`、邻接表图都自己写
- **Python 3.9 兼容**：所有文件 `from __future__ import annotations`；用 `typing.Optional/Dict/List/Tuple`，不用 `X | Y` 或运行时 `list[str]`
- **测试用 `unittest`**（不用 pytest）
- **运行测试**：`cd /home/eg4/pwttt/cdx && python -m unittest discover -s unimem/tests -v`

### 关键设计决策（不要违背）

1. **自定义图而非 networkx**：节点少（3-8 个）+ 边带类型与策略，邻接表足够
2. **槽位 ABC 精简**：基类 4 抽象方法（`write/read/clear/stats`），每个槽位 ABC 仅加 2-3 个
3. **沉淀策略挂在边而非模块**：同一 EM 可向 SM 抽事实、向 GM 抽空间模式
4. **FEEDS 默认恒等传播**：边策略只门控不变换；变换由目标模块 `write()` 内部处理
5. **`MultiAxisIndex` 是工具非强制**：场景图等模块可有自己的树结构
6. **单一 Registry 按 `(slot, impl_name)`**：`list_implementations(slot)` 查询
7. **`GraphSpec` dataclass + `from_dict`**：dataclass 便于 IDE，dict 便于 JSON/YAML 加载

## 关键设计原则（修正研究方向）

1. 框架定义**接口契约**而非具体实现 → 不同方法 plug-in 实现 = 向上兼容
2. 记忆修正**不破坏**原方法功能 → 作为可选中间件 / 模块叠加
3. 每个修正策略需对应**可观测的错误类型**与**可量化的改进信号**
4. 修正的代价（重新感知、计算开销）需与错误造成的下游损失权衡

## 关键参照方法（设计修正机制时对照）

| 方法 | 与记忆修正相关的现有机制 | 局限（修正研究的切入点） |
|------|------------------------|------------------------|
| INHerit-SG | 检索时的 VLM 视觉验证（+8.6%） | 仅在检索时验证候选，非系统性纠错 |
| MemoryEQA | 写入三条件门控 + 熵停止 | 只防新错误进入，不修已有错误 |
| WorldMM | 情景→语义 consolidation（去冲突） | 只在抽象层去重，不回查底层感知 |
| Pred-EQA | 预测-修正闭环（修剪错误预测） | 修正的是规划预测，不是记忆内容 |
| DovSG | 局部子图更新（动态环境） | 只处理物体移动，不处理语义/关系错误 |
| DynaMem | 显式变化检测 | 检测变化但不纠幻觉/错分类 |

## 工作约定

- **语言**：中文为主，技术术语保留英文
- **方法学**：文献驱动；任何框架/机制设计需有论文支撑（可在 papers_matrix.md 追溯）
- **范围**：以 EQA 为主要测试域（有明确 QA 正确性信号便于评估修正效果），向 VLA 通用化
- **代码**：避免过度工程化；不为假设性未来需求设计；不添加未被请求的 docstring/类型注解/特征
- **Git**：commit 前用户明确请求才提交；**不主动推送**（用户会手动 push）
- **避免**：凭空设计无文献支撑的机制；过度增加框架槽位复杂度；
  把框架做成"大而全"却无法服务于修正目标
