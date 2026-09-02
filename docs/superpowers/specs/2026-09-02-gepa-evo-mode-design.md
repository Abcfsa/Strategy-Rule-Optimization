# GEPA 进化模式 + 训练期检索开关 设计文档

**日期**: 2026-09-02
**状态**: 已确认，待写实施计划

## 1. 背景与目标

当前 SRO 的 `train_and_reflect` 只有一套训练流程：`n_iters` 轮全量跑训练集 → 反思 → `_evolve_strategy`（候选打分取最高）→ 更新。这套流程缺乏 GEPA (Generative Pareto Evolution) 的核心机制——Pareto 前沿选择、minibatch 采样、调用预算控制、严格提升接受。

**目标**：
1. 加一个 `EVO_MODE` 开关，开启时用 GEPA 式优化（移植自 `openai_api_test/gepa_aime_v3.py`），关闭时保持现有流程。
2. 加一个 `TRAIN_RETRIEVE_CTX` 开关，控制训练期是否从 KB 检索短期规律作为 TaskLM 的 few-shot 上下文（即 engine.py 第89行的自递归上下文）。

**设计约束**：
- GEPA 模式与 KB（短期规律库）共存，不替代。
- classic 模式（默认）行为与现在完全一致。
- 现有 `reflect()`、`_assemble_system_prompt`（含格式指令）、`extract_answer`、grading 不改。

## 2. 两个开关 + config 参数

### 2.1 开关

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `EVO_MODE` | str (`gepa`/`classic`) | `classic` | 控制训练循环结构 |
| `TRAIN_RETRIEVE_CTX` | bool | `true` | 训练期是否检索 KB 短期规律作 few-shot 上下文 |

`TRAIN_RETRIEVE_CTX` 对 **GEPA 和 classic 模式都生效**：
- `true`：训练期 TaskLM.run 的 `context_examples = kb.retrieve(problem)`（现有行为）
- `false`：训练期 `context_examples = []`（纯靠长期策略，不递归引用已积累的规律）

### 2.2 GEPA 专属参数（仅 gepa 模式生效，classic 忽略）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `MAX_METRIC_CALLS` | int | `150` | API 调用预算，控制 GEPA 主循环终止 |
| `MINIBATCH_SIZE` | int | `8` | 每轮采样的训练样本数 |
| `MAX_PROMPT_LENGTH` | int | `2000` | reflection 输出长度上限 |

### 2.3 config.py 改动

`DEFAULTS` 字典加 5 项；`Config` dataclass 加 5 字段；`Config.load()` 读 5 项。`EVO_MODE` 是字符串直接读；`TRAIN_RETRIEVE_CTX` 用 `_parse_bool`；其余 int。

### 2.4 命令行覆盖

- `--evo-mode {gepa,classic}`（默认从 config）
- `--no-train-ctx`（`dest="train_retrieve_ctx", action="store_false"`，默认从 config）

## 3. Strategy 对象扩展

`llm.py` 的 `Strategy` dataclass 加两个字段：

```python
@dataclass
class Strategy:
    text: str
    score: float = 0.0          # 非 GEPA 用标量；GEPA 用 mean val score
    version: int = 0
    parent_version: int = 0
    val_scores: list[float] = field(default_factory=list)  # GEPA 用，逐样本 val 分数
    parent_idx: Optional[int] = None  # GEPA 候选池索引（候选池里的父）
```

- `val_scores` 默认空列表，非 GEPA 模式不碰，现有代码零影响。
- `parent_idx` 供 GEPA 候选池追踪谱系。

## 4. Pareto 模块（新建 sro/pareto.py）

从 `gepa_aime_v3.py` 移植 4 个函数，约 60 行，改入参类型（v3 用 dict 候选，这里用 `list[Strategy]` 的 `val_scores` 字段）：

```python
def build_pareto_fronts(val_scores_per_candidate: list[list[float]], n_val: int) -> list[set]:
    """每个 val 样本取最高分候选集，返回 fronts。"""

def is_dominated(y, programs, fronts) -> bool: ...

def remove_dominated_programs(fronts, scores=None) -> list[set]: ...

def select_candidate_from_pareto_front(fronts, candidate_scores, rng) -> int:
    """从 Pareto 前沿按频率采样一个候选索引。"""
```

独立文件，不污染 engine.py。

## 5. ReflectionLM 扩展：reflect_gepa

给 `ReflectionLM` 加 `reflect_gepa` 方法，用 v3 式诊断反馈 prompt：

```python
def reflect_gepa(self, parent: Strategy, minibatch_traces: list[Trace],
                 iteration: int) -> str:
    """GEPA 式反思：诊断级反馈 + 错误模式聚类 → 输出新策略文本（整体替换）。
    
    返回新 prompt 文本，调用方清理 markdown 代码块。
    """
```

### 5.1 内部移植的 v3 函数

作为 ReflectionLM 的私有方法或模块函数：
- `_build_diagnostic_feedback(traces) -> str`：每题的完整推理 + pred + gold + 错误分类
- `_cluster_error_patterns(traces) -> str`：跨样本错误模式聚类摘要
- `_classify_error_type(prediction, gold, reasoning) -> str`：错误分类

### 5.2 适配层

- 输入是 `Trace` 列表（SRO 数据结构），适配层转成 `(problem, reasoning, prediction, answer, correct)` 元组。
- v3 的 `_classify_error_type` 是 AIME 专属（format_error / calculation_error / conceptual_error）。移植时对非 AIME 数据集降级为通用分类（只区分 `no_answer_extracted` / `wrong_answer`），避免硬编码 AIME 假设。
- 输出一个 `str`（新策略文本），调用方（engine）负责清理 ``` 代码块、长度校验、与父策略去重。

### 5.3 reflection prompt 文本

移植 v3 的 `build_reflection_prompt` 结构（仿论文 Appendix B 格式），但把 "AIME" 字样改成通用的 "the task"，适配多数据集。

**不改动现有 `reflect()`**——classic 模式照用。两套 reflection 并存在 ReflectionLM 里，由 evo_mode 决定调哪个。

## 6. engine.py 的 GEPA 训练路径

### 6.1 __init__ 新增字段

```python
def __init__(self, ..., 
             evo_mode: str = "classic",
             train_retrieve_ctx: bool = True,
             max_metric_calls: int = 150,
             minibatch_size: int = 8,
             max_prompt_length: int = 2000,
             seed: int = 42):
```

### 6.2 train_and_reflect 分流

```python
def train_and_reflect(self, train_set, n_iters=3, candidates_per_iter=2, verbose=True):
    if self.evo_mode == "gepa":
        return self._train_gepa(train_set, verbose)
    return self._train_classic(train_set, n_iters, candidates_per_iter, verbose)
```

### 6.3 classic 路径（_train_classic）

现有 `train_and_reflect` 的逻辑原样拆出为 `_train_classic`，唯一改动：第89行的 `kb.retrieve` 根据 `train_retrieve_ctx` 决定是否做。

```python
ctx_examples = []
if self.train_retrieve_ctx:
    ctx_examples = self.kb.retrieve(sample.problem, k=self.top_k, threshold=self.match_threshold)
```

### 6.4 GEPA 路径（_train_gepa）

移植 v3 主循环，核心步骤：

```python
def _train_gepa(self, train_set, verbose) -> list[dict]:
    rng = random.Random(self.seed)
    # Pareto 验证子集：从 train 切 1/4 做内部 Pareto 验证（真正的 val 留给最终评测）
    n_val = max(1, len(train_set) // 4)
    val_for_pareto = train_set[:n_val]
    train_pool = train_set[n_val:]  # minibatch 从这里采样
    budget_used = 0
    history = []

    # Step 1: 初始化候选池（初始策略 = TaskLM 当前策略）
    candidates: list[Strategy] = [Strategy(text=self.task_lm.strategy.text or _default_prompt(), version=0)]
    candidates[0].val_scores = self._eval_candidate(candidates[0], val_for_pareto)
    candidates[0].score = sum(candidates[0].val_scores) / n_val
    budget_used += n_val
    pareto_fronts = build_pareto_fronts([c.val_scores for c in candidates], n_val)
    best_idx, best_score = 0, candidates[0].score

    # Step 2: 预算控制主循环
    while budget_used < self.max_metric_calls:
        iteration = len(history) + 1
        # 2a) Pareto 选父
        scores_map = {i: c.score for i, c in enumerate(candidates)}
        parent_idx = select_candidate_from_pareto_front(pareto_fronts, scores_map, rng)
        parent = candidates[parent_idx]
        # 2b) minibatch 采样
        mb_size = min(self.minibatch_size, len(train_pool))
        minibatch = rng.sample(train_pool, mb_size)
        # 2c) 父候选跑 minibatch（带 traces；train_retrieve_ctx 控制检索）
        old_traces = self._run_minibatch(parent, minibatch)
        budget_used += mb_size
        if all(t.result.correct for t in old_traces):
            continue  # 全对跳过
        # 2d) reflection 产新策略（v3 式诊断反馈）
        new_text = self.reflection_lm.reflect_gepa(parent, old_traces, iteration)
        new_text = _clean_markdown(new_text)
        if not new_text or new_text == parent.text:
            continue
        if len(new_text) > self.max_prompt_length:
            continue
        # 2e) 新候选跑同 minibatch
        new_strat = Strategy(text=new_text, version=len(candidates), parent_idx=parent_idx)
        new_traces = self._run_minibatch(new_strat, minibatch)
        budget_used += mb_size
        old_sum = sum(t.result.correct for t in old_traces)
        new_sum = sum(t.result.correct for t in new_traces)
        # 2f) 严格提升才接受
        if new_sum <= old_sum:
            continue
        # 2g) 接受 → 完整 val 评估 → 更新 Pareto
        new_strat.val_scores = self._eval_candidate(new_strat, val_for_pareto)
        budget_used += n_val
        new_strat.score = sum(new_strat.val_scores) / n_val
        candidates.append(new_strat)
        pareto_fronts = build_pareto_fronts([c.val_scores for c in candidates], n_val)
        if new_strat.score > best_score:
            best_idx = len(candidates) - 1
            best_score = new_strat.score
        # 2h) 反思产出的短期规律照常入 KB（KB 共存）
        patterns, _ = self.reflection_lm.reflect(old_traces + new_traces)
        self.kb.add_patterns(patterns)
        history.append({
            "iteration": iteration, "evo_mode": "gepa",
            "parent_idx": parent_idx, "old_minibatch": old_sum,
            "new_minibatch": new_sum, "new_val_score": new_strat.score,
            "accepted": True, "budget_used": budget_used,
            "strategy_version": new_strat.version,
            "strategy_text": new_strat.text,
        })

    # Step 3: 用 best 候选更新 TaskLM
    self.kb.update_strategy(candidates[best_idx])
    self.task_lm.update_strategy(candidates[best_idx])
    return history
```

### 6.5 辅助方法

```python
def _run_minibatch(self, strategy: Strategy, minibatch: list[TrainSample]) -> list[Trace]:
    """临时换上 strategy 跑 minibatch，返回 traces。
    train_retrieve_ctx 控制是否检索 KB。调用方按需从 trace.result.correct 取分数。"""

def _eval_candidate(self, strategy: Strategy, val_subset: list[TrainSample]) -> list[float]:
    """在 val_subset 上跑，返回逐样本 1.0/0.0 分数。"""
```

`_run_minibatch` 和 `_eval_candidate` 都临时换上候选 strategy 跑完恢复原策略（仿现有 `_evolve_strategy` 的做法）。`_eval_candidate` 内部调 `_run_minibatch` 后提取分数列表。

### 6.6 模块级工具函数

```python
def _clean_markdown(text: str) -> str:
    """清理 ``` 代码块包装。"""

def _default_prompt() -> str:
    """GEPA 模式无策略时的初始 prompt（通用，非 AIME 专属）。"""
```

## 7. 输出文件扩展

| 文件 | 改动 |
|---|---|
| `history.json` | 每条记录加 `evo_mode` 字段；GEPA 模式加 `parent_idx`/`old_minibatch`/`new_minibatch`/`budget_used` |
| `summary.json` | GEPA 模式记 `budget_used`/`max_metric_calls`；classic 记 `n_iters` |
| `config.json` | 自动包含 5 个新参数（asdict 自动捕获） |

## 8. 改动文件清单

| 文件 | 改动类型 |
|---|---|
| `sro/llm.py` | Strategy 加字段；ReflectionLM 加 `reflect_gepa` + 诊断反馈方法 |
| `sro/engine.py` | `__init__` 加字段；`train_and_reflect` 分流；新增 `_train_gepa`/`_train_classic`/`_run_minibatch`/`_eval_candidate` |
| `sro/pareto.py` | **新建**，4 个 Pareto 函数 |
| `sro/config.py` | DEFAULTS + dataclass + load 加 5 字段 |
| `main.py` | argparse 加 `--evo-mode`/`--no-train-ctx`；run_dataset 传新参数；_save_outputs 记录 |

**不改**：`knowledge.py`、`datasets.py`、`grading.py`、`_assemble_system_prompt`、`extract_answer`。

## 9. 不变的行为

- `classic` 模式（默认）与现在完全一致——`n_iters` 轮、全量跑、`_evolve_strategy` 取最高分。
- `train_retrieve_ctx=true`（默认）第89行检索照做。
- demo 路程不受影响。
- 所有现有输出文件结构兼容（新字段是增量）。

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| GEPA 候选池初始策略为空导致冷启动差 | `_default_prompt()` 提供通用初始策略（非 AIME 专属） |
| `_classify_error_type` 对非 AIME 数据集不适用 | 降级为通用分类（no_answer / wrong_answer） |
| GEPA 预算耗尽前没接受任何候选 | Step 3 用 best_idx 兜底（至少是初始候选） |
| val_for_pareto 从 train 切走后训练池变小 | 只切 1/4，且 minibatch 从剩余 train_pool 采样 |
| 临时换策略跑 minibatch 竞态 | 仿现有 `_evolve_strategy`，跑完恢复 original_strategy |
