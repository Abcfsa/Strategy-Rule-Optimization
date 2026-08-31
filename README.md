# Strategy-Rule-Optimization (SRO)

两阶段反思式自进化 Prompt 优化框架。Task LM + Reflection LM 双模型，
训练期反思产出短期规律 + 长期策略并迭代，测试期按相似度匹配规律、
未命中时触发动态学习。

```
┌──────────────── 阶段一：训练与反思迭代 ────────────────┐
│  TaskLM(训练集) ──► 轨迹+结果 ──► ReflectionLM 反思   │
│                                          │            │
│            ┌─────────────┴─────────────┐  │            │
│            ▼                          ▼  │            │
│   短期规律(可检索)            长期策略   │            │
│            │                          │  │            │
│            └─────────► 用长期策略迭代 TaskLM prompt ◄──┘
│                        (候选→打分→筛选)  闭环
└──────────────────────────────────────────────────────┘
┌──────────────── 阶段二：测试与推理 ───────────────────┐
│  测试题 ──向量检索──► 短期规律池                        │
│     │                                                  │
│     ├─ 命中 ─► 例子 + 长期策略 ─► TaskLM ─► 回答       │
│     └─ 未命中 ─► 动态学习(待考虑)                       │
│                  临时归纳规律 → 临时入库 → 第二轮测试    │
│                  ─► TaskLM ─► 回答                      │
└──────────────────────────────────────────────────────────┘
```

## 安装

```bash
pip install -e .
```

当前为纯 Python 框架，无第三方硬依赖。

## 使用

```bash
# 占位演示（验证两阶段数据流与分支逻辑）
python main.py --demo
```

## 结构

| 文件 | 内容 |
|------|------|
| `sro/llm.py` | `TaskLM` / `ReflectionLM` / `Embedder` + 数据结构（`Trace`/`Result`/`Example`/`Strategy`） |
| `sro/knowledge.py` | `KnowledgeBase`：短期规律检索 + 长期策略存储 + 临时入库机制 |
| `sro/engine.py` | `SROEngine`：`train_and_reflect()`（阶段一闭环）+ `inference()`（阶段二匹配/不匹配） |
| `main.py` | 演示入口 |

## 接入真实模型

替换 `sro/llm.py` 中各 `_call_llm()` 占位（已用注释标明接入点），
`Embedder.embed()` 替换为真实 embedding（OpenAI / sentence-transformers），
`KnowledgeBase.retrieve()` 可换 FAISS/Chroma。
