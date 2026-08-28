# SRO 设计文档索引

本目录存放开发过程中的设计决策记录。技术路线的完整论证在仓库外部：

- **技术路线报告**：`../../可迁移自进化_技术路线/技术路线报告_可迁移自进化Prompt优化.md`
  - §4.2 三层记忆形式化 → `src/memory/`
  - §4.3 检索机制 → `src/assemble/`
  - §4.4 训练期双向闭环 → `src/evolve/` + `src/distill/`
  - §4.5 测试时快速适应协议 → `src/adapt/`
  - §4.6 条款级正则化（TextReg 融合）→ `src/regularize/`
  - §4.7 第一、二类论文成果接线 → 各模块 docstring 中标注了对应小节
- **前景评估**：`../../可迁移自进化_技术路线/前景评估_认可度与替代路线分析.md`

## 模块 ↔ 报告章节对照

| 模块 | 报告章节 | 主要论文来源 |
|------|---------|-------------|
| `src/memory/` | §4.2 | MemAPO、ERL、Voyager、AdaMEM |
| `src/evolve/` | §4.4, §4.7.1-4.7.2 | GEPA、Toolbox、SEM-Stop、p1、RCL |
| `src/regularize/` | §4.6 | TextReg、RAPOA |
| `src/distill/` | §4.4, §4.7.2 | ETGPO、AIR |
| `src/adapt/` | §4.5, §4.7.3-4.7.4 | Dynamics Grounding、Overthinking、Self-Correction Help |
| `src/assemble/` | §4.3 | ERL、2307.07164、AgentBench |
| `src/dataset/` | §5.1 | 复用 gepa_aime.py / gepa_math.py |

## 开发约定

- 数据加载、答案解析、API 调用等已验证代码从 `../openai_api_test/` 移植，
  移植来源在各模块 docstring 中注明
- 每个模块实现前先对照报告对应章节与模块 docstring 中的机制描述
- 实验编号（E1-E11）见报告 §5.3，结果存放于 `outputs/`（gitignore）
