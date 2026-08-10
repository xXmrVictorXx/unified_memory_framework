# 通用 EQA 记忆框架 - 调研存档

> 调研主题：通用具身问答（EQA）智能体记忆框架设计
> 起始日期：2026-08-10
> 累计论文：~55 篇（去重）

## 文件索引

| 文件 | 内容 | 用途 |
|------|------|------|
| `research_notes.md` | **研究结论存档（首先读这个）** | 跨对话延续上下文：目标、发现、框架设计、开放问题 |
| `papers_matrix.md` | 论文 × 记忆模块分类矩阵 + 融合模式 | 写 Related Work 时按记忆类型检索；可视化热力图 |
| `references.bib` | 完整 BibTeX 文献库 | 论文写作直接引用；按类别分组；`fusion=` 标签可 grep |

## 快速入口

- **新对话开始时**：先读 `research_notes.md` 第 1、3、4、7 节
- **找某记忆类型的论文**：查 `papers_matrix.md` 第二节矩阵
- **找某创新概念的引用**：查 `papers_matrix.md` 第六节
- **写论文引文**：在 `references.bib` 中按类别或 `fusion=` 标签 grep

## 核实中的几处重要修正（相对原始调研）

以下修正来自第二轮 BibTeX 核实任务，已反映在 `references.bib` 中：

1. **ConceptGraphs** 一作是 **Qiao Gu**（非 Rosen）；会议为 RSS 2024
2. **Hydra** 会议为 **RSS 2022**（非 RAL）
3. **OpenEQA** arXiv ID 为 **2408.12282**（DBLP 索引；2406.09701 疑为早期版本）
4. **K-EQA** 原预印 2021，发表于 **TPAMI 2023**
5. **SAT-AMA** 全称 "Spatially-Aware Transformer for Embodied Agents"（SAT + Adaptive Memory Allocator）
6. **EMQA** 作者是 Samyak Datta & Sameer Dharur（非 Majumdar/Yadav，后者属 OpenEQA）
7. **Episodic eKG** 未能定位（可能改名或未被索引），已从 bib 排除，待人工查证
8. **M4EI / R-EQA / M2PA / 3DGraphQA / EpiMem** 无 arXiv，用 venue/DOI 引用

## 2025-2026 多记忆融合论文（用户重点关注）

| 论文 | arXiv | 核心融合创新 |
|------|-------|------------|
| R4 | 2512.15940 | 对象级三轴索引(语义+空间+时间) |
| Pred-EQA | CVPR 2026 | 文本控制面 + 视觉证据数据面 |
| INHerit-SG | 2602.12971 | 4层层级图 + 视角指针 + 双流异步 |
| WorldMM | 2512.02425 | 多时间尺度情景 + 演化语义 + 自适应检索Agent |
| MemoryEQA | 2505.13948 | 熵自适应检索 + 多模块分发 |
| RoboMemory | 2508.01415 | 4模块并行 + critic仲裁 |
| BrainMem | 2604.16331 | 自动符号规则沉淀 |

## 下一步候选方向

见 `research_notes.md` 第 8 节。
