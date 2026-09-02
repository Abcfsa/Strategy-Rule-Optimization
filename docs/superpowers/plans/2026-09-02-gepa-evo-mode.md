# GEPA 进化模式 + 训练期检索开关 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 SRO 加一个 GEPA 式进化模式开关（Pareto 前沿 + minibatch + 预算 + keep-better），和一个训练期 KB 检索开关，两者独立可组合。

**Architecture:** `train_and_reflect` 按 `evo_mode` 分流到 `_train_classic`（现有逻辑）或 `_train_gepa`（移植 gepa_aime_v3.py 主循环）。GEPA 模式与 KB 共存——候选策略走 Pareto 进化，短期规律照常入 KB。Pareto 函数独立到新模块 `sro/pareto.py`。

**Tech Stack:** Python 3.10+, dataclasses, openai SDK, sentence-transformers。无外部测试框架——用 `python -m py_compile` 编译检查 + 手动冒烟。

**Spec:** `docs/superpowers/specs/2026-09-02-gepa-evo-mode-design.md`

## Global Constraints

- 项目非 git 仓库（`Is a git repository: false`），无 commit 步骤；用阶段性 `python -m py_compile` 编译检查替代。
- 用户红线：改动代码后，用户同意后才跑测试/验证命令。每个任务末尾的编译检查需用户同意后执行。
- 所有输入 LLM 的文本用英文（用户要求）。
- 密钥不入代码、不写盘。`config.json` 用 `asdict(cfg)` 后 `pop("openai_api_key")`。
- 现有 `reflect()`、`_assemble_system_prompt`、`extract_answer`、`grading.py`、`knowledge.py`、`datasets.py` 不改。
- `classic` 模式（默认）行为与现在完全一致。

---

## File Structure

| 文件 | 职责 | 改动类型 |
|---|---|---|
| `sro/pareto.py` | 4 个 Pareto 前沿函数（移植自 v3） | 新建 |
| `sro/llm.py` | Strategy 扩展字段；ReflectionLM 加 reflect_gepa + 诊断反馈 | 修改 |
| `sro/engine.py` | train_and_reflect 分流；_train_gepa/_train_classic/辅助方法 | 修改 |
| `sro/config.py` | 5 个新 config 字段 | 修改 |
| `sro/__init__.py` | 导出 pareto 模块 | 修改 |
| `main.py` | argparse 加 --evo-mode/--no-train-ctx；run_dataset 传参；_save_outputs 扩展 | 修改 |
| `.env.example` | 5 个新参数示例 | 修改 |

---

### Task 1: 新建 sro/pareto.py — Pareto 前沿函数

**Files:**
- Create: `sro/pareto.py`

**Interfaces:**
- Produces: `build_pareto_fronts(val_scores_per_candidate, n_val) -> list[set]`、`is_dominated(y, programs, fronts) -> bool`、`remove_dominated_programs(fronts, scores=None) -> list[set]`、`select_candidate_from_pareto_front(fronts, candidate_scores, rng) -> int`

- [ ] **Step 1: 写 sro/pareto.py**

移植自 `gepa_aime_v3.py` 的四个函数，入参从 dict 候选改为索引 + val_scores 列表。逻辑原样保留：

```python
"""Pareto 前沿选择（移植自 gepa_aime_v3.py）。

候选策略池用索引（int）标识；每个候选的 val 表现用逐样本 1.0/0.0 分数列表表示。
Pareto 前沿 = 每个 val 样本上得分最高的候选集合。
"""

from __future__ import annotations

from typing import Optional


def build_pareto_fronts(val_scores_per_candidate: list[list[float]],
                        n_val: int) -> list[set]:
    """每个 val 样本取最高分候选集，返回 n_val 个前沿集合。"""
    fronts = []
    for i in range(n_val):
        scores_i = [s[i] for s in val_scores_per_candidate]
        best_score = max(scores_i)
        front = {j for j, s in enumerate(scores_i) if s == best_score}
        fronts.append(front)
    return fronts


def is_dominated(y: int, programs: set, fronts: list[set]) -> bool:
    """y 是否被 programs 中的某个候选在所有 y 所属前沿上支配。"""
    y_fronts = [front for front in fronts if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    return True


def remove_dominated_programs(fronts: list[set],
                              scores: Optional[dict] = None) -> list[set]:
    """移除被支配的候选，返回清洗后的前沿列表。"""
    freq = {}
    for front in fronts:
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())
    if scores is None:
        scores = dict.fromkeys(programs, 1)

    programs = sorted(programs, key=lambda x: scores[x], reverse=False)

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), fronts):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs if p not in dominated]
    new_fronts = [
        {prog_idx for prog_idx in front if prog_idx in dominators}
        for front in fronts
    ]
    return new_fronts


def select_candidate_from_pareto_front(fronts: list[set],
                                       candidate_scores: dict,
                                       rng) -> int:
    """从 Pareto 前沿按频率加权采样一个候选索引。"""
    new_fronts = remove_dominated_programs(fronts, scores=candidate_scores)

    program_frequency = {}
    for front in new_fronts:
        for prog_idx in front:
            program_frequency[prog_idx] = program_frequency.get(prog_idx, 0) + 1

    sampling_list = []
    for prog_idx, freq in program_frequency.items():
        sampling_list.extend([prog_idx] * freq)

    assert len(sampling_list) > 0, "Pareto front is empty!"
    return int(rng.choice(sampling_list))
```

- [ ] **Step 2: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/pareto.py`
Expected: 无输出（编译通过）

- [ ] **Step 3: 在 sro/__init__.py 导出**

读 `sro/__init__.py`，在现有 import 块后加一行：

```python
from . import pareto  # noqa: F401
```

- [ ] **Step 4: 编译检查 __init__**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/__init__.py sro/pareto.py`
Expected: 无输出

---

### Task 2: Strategy 对象扩展（llm.py）

**Files:**
- Modify: `sro/llm.py`（Strategy dataclass，约第68-76行）

**Interfaces:**
- Consumes: 无
- Produces: `Strategy` 新增 `val_scores: list[float]` 和 `parent_idx: Optional[int]` 字段，默认空列表/None

- [ ] **Step 1: 读当前 Strategy 定义**

Run: 读 `sro/llm.py` 第68-76行附近，确认现有字段。

- [ ] **Step 2: 扩展 Strategy**

把现有的：

```python
@dataclass
class Strategy:
    """长期策略（System Prompt 级别的高级方法论）。"""

    text: str                             # 策略正文
    score: float = 0.0                    # 该策略在验证集上的得分
    version: int = 0                      # 迭代版本号
    parent_version: int = 0              # 由哪个版本演化而来
```

改成：

```python
@dataclass
class Strategy:
    """长期策略（System Prompt 级别的高级方法论）。

    GEPA 模式下 val_scores 携带逐样本验证分数（供 Pareto 前沿用），
    parent_idx 追踪候选池谱系；非 GEPA 模式不碰这两个字段。
    """

    text: str                             # 策略正文
    score: float = 0.0                    # 该策略在验证集上的得分（GEPA 用 mean）
    version: int = 0                      # 迭代版本号
    parent_version: int = 0              # 由哪个版本演化而来
    val_scores: list[float] = field(default_factory=list)  # GEPA 逐样本 val 分数
    parent_idx: Optional[int] = None      # GEPA 候选池父索引
```

注意：`field` 和 `Optional` 已在 llm.py 顶部 import（`from dataclasses import dataclass, field` 和 `from typing import Any, Optional`）——确认这两行存在。

- [ ] **Step 3: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/llm.py`
Expected: 无输出

---

### Task 3: ReflectionLM 加 reflect_gepa + 诊断反馈（llm.py）

**Files:**
- Modify: `sro/llm.py`（ReflectionLM 类，约第535行起）

**Interfaces:**
- Consumes: `Trace`（现有）、`Strategy`（Task 2 扩展后的）
- Produces: `ReflectionLM.reflect_gepa(parent, minibatch_traces, iteration) -> str`；私有方法 `_build_diagnostic_feedback(traces) -> str`、`_cluster_error_patterns(traces) -> str`、`_classify_error_type(prediction, gold, reasoning) -> str`、`_count_reasoning_steps(reasoning) -> int`

- [ ] **Step 1: 读 ReflectionLM 类范围**

读 `sro/llm.py` 从 ReflectionLM 类定义到文件末尾，确认 `reflect` 和 `extract_pattern_from_question` 的位置。

- [ ] **Step 2: 加 _count_reasoning_steps 和 _classify_error_type（通用化）**

在 ReflectionLM 类内、`reflect` 方法之前加两个 @staticmethod。注意：v3 的 `_classify_error_type` 是 AIME 专属（检查 0-999 整数范围）。这里通用化——对非 AIME 只区分 `no_answer_extracted` / `wrong_answer`：

```python
    @staticmethod
    def _count_reasoning_steps(reasoning: str) -> int:
        """Roughly count reasoning steps (by sentence / newline / markers)."""
        if not reasoning:
            return 0
        import re
        segments = re.split(
            r'[.\n]|(?:Step\s*\d+)|(?:First|Second|Next|Finally|Therefore)',
            reasoning, flags=re.IGNORECASE)
        return max(1, len([s for s in segments if s.strip()]))

    @staticmethod
    def _classify_error_type(prediction: str, ground_truth: str,
                             reasoning: str) -> str:
        """Generic error classification (dataset-agnostic).

        v3's version is AIME-specific (checks 0-999 integer range); here we
        degrade to a simple two-way split for non-AIME datasets, avoiding
        hardcoded AIME assumptions in the general flow.
        """
        if not prediction or not prediction.strip():
            return "no_answer_extracted"
        return "wrong_answer"
```

- [ ] **Step 3: 加 _build_diagnostic_feedback**

在 ReflectionLM 类内加。适配层：v3 接收 dict 列表（`{problem, reasoning, prediction, answer, correct}`），这里接收 `list[Trace]`，转成同样的字段访问：

```python
    def _build_diagnostic_feedback(self, traces: list[Trace]) -> str:
        """Build diagnostic feedback text from minibatch traces.

        Adapts v3's _build_diagnostic_feedback to SRO's Trace objects.
        """
        lines = []
        for i, t in enumerate(traces):
            status = "CORRECT" if t.result.correct else "WRONG"
            error_type = "" if t.result.correct else self._classify_error_type(
                t.result.answer, "", t.trajectory)
            entry = (
                f"--- Example {i+1} [{status}]"
                f"{f'  Error type: {error_type}' if error_type else ''}\n"
                f"Problem: {t.problem}\n"
                f"Model's full reasoning:\n{t.trajectory}\n"
                f"Model's answer: {t.result.answer}\n"
                f"Correct answer: {t.context.get('gold_answer', 'N/A')}\n"
            )
            if not t.result.correct:
                entry += (
                    f"Diagnosis: The model produced {error_type.replace('_', ' ')}. "
                    f"Reasoning length: {self._count_reasoning_steps(t.trajectory)} estimated steps. "
                    f"Got '{t.result.answer}'.\n"
                )
            lines.append(entry)
        return "\n".join(lines)
```

注意：`t.context` 是 Trace 的 dict 字段（`{"strategy_version", "n_examples", "examples_text", "answer_type"}`）。gold_answer 不在其中——下面 Task 4 的 `_run_minibatch` 会把 gold_answer 注入 context。如果 context 里没有，这里用 `'N/A'` 兜底（`context.get('gold_answer', 'N/A')`）。

- [ ] **Step 4: 加 _cluster_error_patterns**

```python
    def _cluster_error_patterns(self, traces: list[Trace]) -> str:
        """Cross-sample error pattern clustering summary."""
        errors = [t for t in traces if not t.result.correct]
        if not errors:
            return "No errors in this batch."

        error_types = {}
        for t in errors:
            et = self._classify_error_type(t.result.answer, "", t.trajectory)
            error_types[et] = error_types.get(et, 0) + 1

        summary_lines = [f"Error pattern summary ({len(errors)} wrong out of {len(traces)}):"]
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  - {et.replace('_', ' ')}: {count} case(s)")
        return "\n".join(summary_lines)
```

- [ ] **Step 5: 加 reflect_gepa**

在 ReflectionLM 类内、`extract_pattern_from_question` 之前加：

```python
    def reflect_gepa(self, parent: Strategy, minibatch_traces: list[Trace],
                     iteration: int) -> str:
        """GEPA-style reflection: diagnostic feedback + error clustering -> new strategy text.

        Returns new prompt text (complete replacement). Caller cleans markdown
        fences and validates length. Uses v3's build_reflection_prompt structure
        but generalizes 'AIME' to 'the task' for multi-dataset support.
        """
        detailed_traces = self._build_diagnostic_feedback(minibatch_traces)
        error_summary = self._cluster_error_patterns(minibatch_traces)
        reflection_prompt = (
            f"I provided an assistant with the following instruction to solve the task:\n\n"
            f"```\n{parent.text}\n```\n\n"
            f"The following are examples of different task inputs provided to the assistant\n"
            f"along with the assistant's response for each of them, and some feedback on\n"
            f"how the assistant's response could be better:\n\n"
            f"```\n{detailed_traces}\n```\n\n"
            f"---\n"
            f"Error pattern summary across the batch:\n"
            f"{error_summary}\n"
            f"---\n\n"
            f"Your task is to write a new instruction for the assistant.\n\n"
            f"Analyze the execution traces and feedback above:\n"
            f"1. Identify the root causes of failures — are they reasoning gaps, format issues,\n"
            f"   or missing verification steps?\n"
            f"2. Note any successful strategies the assistant used on correct examples.\n"
            f"3. Include any generalizable problem-solving heuristics that the current\n"
            f"   instruction omits.\n"
            f"4. Propose strategies the assistant can apply to unseen problems.\n\n"
            f"Provide the new instruction within ``` blocks.\n"
            f"The new instruction must be a complete replacement (not a diff or patch).\n"
            f"Do NOT include explanations — only the new instruction."
        )
        sys = ("You are an expert prompt engineer. Return ONLY the new instruction text.")
        return self._call_llm(system=sys, user=reflection_prompt)
```

- [ ] **Step 6: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/llm.py`
Expected: 无输出

---

### Task 4: engine.py __init__ 加新字段 + 模块级工具函数

**Files:**
- Modify: `sro/engine.py`（`__init__` + 文件顶部加模块级函数）

**Interfaces:**
- Consumes: config 字段（Task 5 加）
- Produces: `SROEngine.__init__` 新参数 `evo_mode`/`train_retrieve_ctx`/`max_metric_calls`/`minibatch_size`/`max_prompt_length`/`seed`；模块函数 `_clean_markdown(text)`、`_default_prompt()`

- [ ] **Step 1: 读 engine.py 顶部和 __init__**

读 `sro/engine.py` 第1-45行，确认 import 和 `__init__` 现状。

- [ ] **Step 2: 加模块级工具函数**

在 `SROEngine` 类定义之前（import 块之后）加：

```python
def _clean_markdown(text: str) -> str:
    """Strip ``` code fence wrapping from reflection output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _default_prompt() -> str:
    """GEPA mode initial prompt when no strategy is set (generic, not AIME-specific)."""
    return (
        "You are an expert problem solver. Read the problem carefully, reason "
        "step by step, and produce the final answer in the format specified by "
        "the task. Double-check your answer before outputting it."
    )
```

- [ ] **Step 3: 扩展 __init__**

把现有的 `__init__`（含上一轮加的 `dynamic_learning`）扩展。在 `dynamic_learning: bool = True` 之后加新参数：

```python
    def __init__(
        self,
        task_lm: Optional[TaskLM] = None,
        reflection_lm: Optional[ReflectionLM] = None,
        embedder: Optional[Embedder] = None,
        kb: Optional[KnowledgeBase] = None,
        match_threshold: float = 0.6,
        top_k: int = 3,
        dynamic_learning: bool = True,
        evo_mode: str = "classic",
        train_retrieve_ctx: bool = True,
        max_metric_calls: int = 150,
        minibatch_size: int = 8,
        max_prompt_length: int = 2000,
        seed: int = 42,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.task_lm = task_lm or TaskLM(self.embedder)
        self.reflection_lm = reflection_lm or ReflectionLM(self.embedder)
        self.kb = kb or KnowledgeBase(self.embedder)
        self.match_threshold = match_threshold
        self.top_k = top_k
        self.dynamic_learning = dynamic_learning
        self.evo_mode = evo_mode
        self.train_retrieve_ctx = train_retrieve_ctx
        self.max_metric_calls = max_metric_calls
        self.minibatch_size = minibatch_size
        self.max_prompt_length = max_prompt_length
        self.seed = seed
```

- [ ] **Step 4: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/engine.py`
Expected: 无输出（此时 train_and_reflect 还没分流，但编译不报错）

---

### Task 5: config.py 加 5 个新字段

**Files:**
- Modify: `sro/config.py`（DEFAULTS + Config dataclass + load）

**Interfaces:**
- Produces: `Config.evo_mode`/`train_retrieve_ctx`/`max_metric_calls`/`minibatch_size`/`max_prompt_length`

- [ ] **Step 1: 加 DEFAULTS 条目**

在 DEFAULTS 字典的 `DYNAMIC_LEARNING` 之后加：

```python
    "DYNAMIC_LEARNING": "true",
    # ---- GEPA 进化模式参数 ----
    "EVO_MODE": "classic",
    "TRAIN_RETRIEVE_CTX": "true",
    "MAX_METRIC_CALLS": "150",
    "MINIBATCH_SIZE": "8",
    "MAX_PROMPT_LENGTH": "2000",
```

- [ ] **Step 2: 加 Config dataclass 字段**

在 `dynamic_learning: bool` 之后加：

```python
    dynamic_learning: bool
    evo_mode: str
    train_retrieve_ctx: bool
    max_metric_calls: int
    minibatch_size: int
    max_prompt_length: int
```

- [ ] **Step 3: 加 load() 读取**

在 `dynamic_learning=_parse_bool(g("DYNAMIC_LEARNING")),` 之后加：

```python
            evo_mode=g("EVO_MODE"),
            train_retrieve_ctx=_parse_bool(g("TRAIN_RETRIEVE_CTX")),
            max_metric_calls=int(g("MAX_METRIC_CALLS")),
            minibatch_size=int(g("MINIBATCH_SIZE")),
            max_prompt_length=int(g("MAX_PROMPT_LENGTH")),
```

- [ ] **Step 4: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/config.py`
Expected: 无输出

---

### Task 6: engine.py train_and_reflect 分流 + _train_classic

**Files:**
- Modify: `sro/engine.py`（train_and_reflect 改为分流；现有逻辑拆出为 _train_classic）

**Interfaces:**
- Consumes: `self.evo_mode`/`self.train_retrieve_ctx`（Task 4）
- Produces: `train_and_reflect` 分流；`_train_classic` 现有逻辑 + train_retrieve_ctx 开关

- [ ] **Step 1: 读当前 train_and_reflect**

读 `sro/engine.py` 第62-136行（现有 train_and_reflect 全部）。

- [ ] **Step 2: 改 train_and_reflect 为分流 + 拆出 _train_classic**

把现有 `train_and_reflect` 方法体改为分流：

```python
    def train_and_reflect(
        self,
        train_set: list[TrainSample],
        n_iters: int = 3,
        candidates_per_iter: int = 2,
        verbose: bool = True,
    ) -> list[dict]:
        """训练闭环。按 evo_mode 分流。返回每轮历史记录。"""
        if self.evo_mode == "gepa":
            return self._train_gepa(train_set, verbose)
        return self._train_classic(train_set, n_iters, candidates_per_iter, verbose)
```

然后把原来的方法体（history 循环）拆成一个新方法 `_train_classic`，唯一改动：第89行的 `kb.retrieve` 加 `train_retrieve_ctx` 开关：

```python
    def _train_classic(
        self,
        train_set: list[TrainSample],
        n_iters: int,
        candidates_per_iter: int,
        verbose: bool,
    ) -> list[dict]:
        """Classic training loop: full-train run -> reflect -> evolve strategy."""
        history: list[dict] = []
        for it in range(1, n_iters + 1):
            if verbose:
                print(f"\n=== Train iteration {it}/{n_iters} ===")

            traces: list[Trace] = []
            for sample in train_set:
                ctx_examples = []
                if self.train_retrieve_ctx:
                    ctx_examples = self.kb.retrieve(
                        sample.problem, k=self.top_k, threshold=self.match_threshold
                    )
                trace = self.task_lm.run(
                    sample.problem,
                    context_examples=ctx_examples,
                    gold_answer=sample.answer,
                    answer_type=sample.answer_type,
                )
                traces.append(trace)

            short_patterns, long_strategy = self.reflection_lm.reflect(traces)

            patterns_before = len(self.kb.examples)
            self.kb.add_patterns(short_patterns)
            patterns_after = len(self.kb.examples)

            best_strategy = self._evolve_strategy(
                long_strategy, traces, train_set, candidates_per_iter
            )
            self.kb.update_strategy(best_strategy)
            self.task_lm.update_strategy(best_strategy)

            acc = (sum(t.result.correct for t in traces) / len(traces)
                   if traces else 0.0)
            record = {
                "iteration": it, "evo_mode": "classic",
                "accuracy": acc,
                "new_patterns": len(short_patterns),
                "patterns_total": patterns_after,
                "patterns_added": patterns_after - patterns_before,
                "strategy_version": best_strategy.version,
                "strategy_score": best_strategy.score,
                "strategy_text": best_strategy.text,
            }
            history.append(record)

            if verbose:
                print(f"  Accuracy this round: {acc:.2%}")
                print(f"  New patterns: {len(short_patterns)}"
                      f" (total {len(self.kb.examples)})")
                print(f"  Strategy version: v{best_strategy.version}")
        return history
```

注意：record 加了 `"evo_mode": "classic"` 字段（之前没有）。

- [ ] **Step 3: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/engine.py`
Expected: 无输出（此时 _train_gepa 还没加，但 train_and_reflect 引用了它——编译不报错因为 Python 延迟解析方法）

---

### Task 7: engine.py 加 _train_gepa + 辅助方法

**Files:**
- Modify: `sro/engine.py`（在 _train_classic 之后加 _train_gepa、_run_minibatch、_eval_candidate）

**Interfaces:**
- Consumes: `pareto.build_pareto_fronts`/`select_candidate_from_pareto_front`（Task 1）、`ReflectionLM.reflect_gepa`（Task 3）、`_clean_markdown`/`_default_prompt`（Task 4）、`Strategy.val_scores`（Task 2）
- Produces: `_train_gepa`/`_run_minibatch`/`_eval_candidate`

- [ ] **Step 1: 在 _train_classic 之后加辅助方法 _run_minibatch 和 _eval_candidate**

```python
    def _run_minibatch(self, strategy: Strategy,
                       minibatch: list[TrainSample]) -> list[Trace]:
        """Temporarily swap in strategy, run minibatch, restore. Returns traces.

        train_retrieve_ctx controls whether KB patterns are retrieved as context.
        gold_answer is injected into trace.context for diagnostic feedback.
        """
        original = self.task_lm.strategy
        self.task_lm.update_strategy(strategy)
        traces: list[Trace] = []
        for sample in minibatch:
            ctx_examples = []
            if self.train_retrieve_ctx:
                ctx_examples = self.kb.retrieve(
                    sample.problem, k=self.top_k, threshold=self.match_threshold
                )
            trace = self.task_lm.run(
                sample.problem,
                context_examples=ctx_examples,
                gold_answer=sample.answer,
                answer_type=sample.answer_type,
            )
            # inject gold_answer into context for reflect_gepa diagnostic feedback
            trace.context["gold_answer"] = sample.answer
            traces.append(trace)
        self.task_lm.update_strategy(original)
        return traces

    def _eval_candidate(self, strategy: Strategy,
                        val_subset: list[TrainSample]) -> list[float]:
        """Run strategy on val_subset, return per-sample 1.0/0.0 scores."""
        original = self.task_lm.strategy
        self.task_lm.update_strategy(strategy)
        scores: list[float] = []
        for sample in val_subset:
            trace = self.task_lm.run(
                sample.problem,
                context_examples=[],  # val eval: no KB context, pure strategy
                gold_answer=sample.answer,
                answer_type=sample.answer_type,
            )
            scores.append(1.0 if trace.result.correct else 0.0)
        self.task_lm.update_strategy(original)
        return scores
```

注意：`_eval_candidate` 不检索 KB（`context_examples=[]`），因为这是评估策略本身的表现，不是推理时使用。这与 v3 的 `evaluate_candidate_on_samples` 一致。

- [ ] **Step 2: 加 _train_gepa 主体**

在辅助方法之后加。这是核心移植，逻辑严格对应 spec Section 6.4：

```python
    def _train_gepa(self, train_set: list[TrainSample],
                    verbose: bool) -> list[dict]:
        """GEPA-style training: Pareto front + minibatch + budget + keep-better.

        Mirrors gepa_aime_v3.py main loop. KB coexists: short patterns from
        reflection are added to KB alongside the Pareto candidate pool.
        """
        import random
        from .pareto import build_pareto_fronts, select_candidate_from_pareto_front

        rng = random.Random(self.seed)
        # Pareto validation subset: cut 1/4 of train for internal Pareto validation.
        # The real val split is reserved for final evaluation in main.py.
        n_val = max(1, len(train_set) // 4)
        val_for_pareto = train_set[:n_val]
        train_pool = train_set[n_val:]
        if not train_pool:
            train_pool = list(train_set)  # degenerate: all used as both
        budget_used = 0
        history: list[dict] = []

        # Step 1: initialize candidate pool
        init_text = self.task_lm.strategy.text or _default_prompt()
        candidates: list[Strategy] = [Strategy(text=init_text, version=0)]
        candidates[0].val_scores = self._eval_candidate(candidates[0], val_for_pareto)
        candidates[0].score = sum(candidates[0].val_scores) / n_val
        budget_used += n_val
        pareto_fronts = build_pareto_fronts(
            [c.val_scores for c in candidates], n_val)
        best_idx, best_score = 0, candidates[0].score

        if verbose:
            print(f"[Init] Base candidate val score: {best_score:.2%}"
                  f" ({n_val} calls)")
            print(f"[Budget] Used: {budget_used}/{self.max_metric_calls}")

        # Step 2: budget-controlled main loop
        while budget_used < self.max_metric_calls:
            iteration = len(history) + 1
            if verbose:
                print(f"\n=== GEPA iteration {iteration} (budget {budget_used}/{self.max_metric_calls}) ===")

            try:
                # 2a) Pareto select parent
                scores_map = {i: c.score for i, c in enumerate(candidates)}
                parent_idx = select_candidate_from_pareto_front(
                    pareto_fronts, scores_map, rng)
                parent = candidates[parent_idx]
                if verbose:
                    print(f"[Select] Parent: candidate #{parent_idx}"
                          f" (val={parent.score:.2%})")

                # 2b) minibatch sample
                mb_size = min(self.minibatch_size, len(train_pool))
                minibatch = rng.sample(train_pool, mb_size)

                # 2c) parent runs minibatch (with traces)
                old_traces = self._run_minibatch(parent, minibatch)
                budget_used += mb_size
                old_sum = sum(t.result.correct for t in old_traces)
                if all(t.result.correct for t in old_traces):
                    if verbose:
                        print("[Skip] All minibatch correct, skipping.")
                    continue
                if verbose:
                    print(f"[Eval] Parent minibatch: {old_sum}/{len(minibatch)}"
                          f" ({mb_size} calls)")

                # 2d) reflection -> new strategy text
                new_text = self.reflection_lm.reflect_gepa(
                    parent, old_traces, iteration)
                new_text = _clean_markdown(new_text)
                if not new_text or new_text == parent.text:
                    if verbose:
                        print("[Reject] Reflection empty or identical.")
                    continue
                if len(new_text) > self.max_prompt_length:
                    if verbose:
                        print(f"[Reject] Too long ({len(new_text)}"
                              f" > {self.max_prompt_length}).")
                    continue

                # 2e) new candidate runs same minibatch
                new_strat = Strategy(
                    text=new_text, version=len(candidates),
                    parent_idx=parent_idx)
                new_traces = self._run_minibatch(new_strat, minibatch)
                budget_used += mb_size
                new_sum = sum(t.result.correct for t in new_traces)

                # 2f) strict improvement acceptance
                if new_sum <= old_sum:
                    if verbose:
                        print(f"[Reject] No improvement ({new_sum} <= {old_sum}).")
                    continue

                # 2g) accept -> full val eval -> update Pareto
                new_strat.val_scores = self._eval_candidate(
                    new_strat, val_for_pareto)
                budget_used += n_val
                new_strat.score = sum(new_strat.val_scores) / n_val
                candidates.append(new_strat)
                pareto_fronts = build_pareto_fronts(
                    [c.val_scores for c in candidates], n_val)
                if verbose:
                    print(f"[Accept] Candidate #{len(candidates)-1}"
                          f" val={new_strat.score:.2%}")
                if new_strat.score > best_score:
                    best_idx = len(candidates) - 1
                    best_score = new_strat.score
                    if verbose:
                        print(f"  * New best! ({best_score:.2%})")

                # 2h) short patterns from reflection -> KB (coexistence)
                patterns, _ = self.reflection_lm.reflect(old_traces + new_traces)
                self.kb.add_patterns(patterns)

                history.append({
                    "iteration": iteration, "evo_mode": "gepa",
                    "parent_idx": parent_idx,
                    "old_minibatch": old_sum, "new_minibatch": new_sum,
                    "new_val_score": new_strat.score,
                    "accepted": True, "budget_used": budget_used,
                    "strategy_version": new_strat.version,
                    "strategy_text": new_strat.text,
                })

            except Exception as e:
                if verbose:
                    print(f"[Error] iteration {iteration}: {e}")
                    import traceback
                    traceback.print_exc()
                break

        # Step 3: best candidate updates TaskLM
        best = candidates[best_idx]
        self.kb.update_strategy(best)
        self.task_lm.update_strategy(best)
        if verbose:
            print(f"\n[Done] Best candidate #{best_idx}"
                  f" val={best_score:.2%}, budget {budget_used}/{self.max_metric_calls}")
        return history
```

- [ ] **Step 3: 编译检查**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && python -m py_compile sro/engine.py`
Expected: 无输出

---

### Task 8: main.py argparse + run_dataset + _save_outputs 扩展

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `Config` 新字段（Task 5）、`SROEngine` 新参数（Task 4）
- Produces: `--evo-mode`/`--no-train-ctx` CLI；run_dataset 传新参数；_save_outputs 记录 evo_mode

- [ ] **Step 1: 加 argparse 参数**

在 `main()` 的 argparse 里，`--no-dynamic` 之后加：

```python
    parser.add_argument("--evo-mode", choices=["gepa", "classic"],
                        default=cfg.evo_mode,
                        help=f"evolution mode (default from .env: {cfg.evo_mode})")
    parser.add_argument("--train-retrieve-ctx", action="store_true",
                        default=cfg.train_retrieve_ctx,
                        help="retrieve KB patterns during training (default from .env)")
    parser.add_argument("--no-train-ctx", dest="train_retrieve_ctx",
                        action="store_false",
                        help="disable KB retrieval during training")
```

- [ ] **Step 2: run_dataset 签名加新参数 + 传给 SROEngine**

把 `run_dataset` 签名从：

```python
def run_dataset(
    dataset: str, n_train: int, n_val: int, n_iters: int,
    seed: int, dynamic_learning: bool, output_dir: str | None,
) -> None:
```

改为：

```python
def run_dataset(
    dataset: str, n_train: int, n_val: int, n_iters: int,
    seed: int, dynamic_learning: bool, output_dir: str | None,
    evo_mode: str, train_retrieve_ctx: bool,
) -> None:
```

在 `engine = SROEngine(...)` 处加新参数：

```python
    engine = SROEngine(
        match_threshold=cfg.match_threshold, top_k=cfg.top_k,
        dynamic_learning=dynamic_learning,
        evo_mode=evo_mode, train_retrieve_ctx=train_retrieve_ctx,
        max_metric_calls=cfg.max_metric_calls, minibatch_size=cfg.minibatch_size,
        max_prompt_length=cfg.max_prompt_length, seed=seed,
    )
```

- [ ] **Step 3: run_params 加新字段**

在 `run_params = {...}` 字典里加：

```python
    run_params = {
        "n_train": n_train, "n_val": n_val, "n_iters": n_iters,
        "seed": seed, "dynamic_learning": dynamic_learning,
        "evo_mode": evo_mode, "train_retrieve_ctx": train_retrieve_ctx,
    }
```

- [ ] **Step 4: _save_outputs 的 summary 改造（兼容两种模式）**

现有 [main.py:81-86](main.py#L81-L86) 的 `iterations` 推导式硬取 `h["accuracy"]`/`h["new_patterns"]`/`h["patterns_total"]`——这些字段只在 classic 模式的 history 记录里有。GEPA 模式的 history 字段是 `parent_idx`/`old_minibatch`/`new_minibatch`/`new_val_score`/`budget_used`，直接复用会 KeyError。改成 `.get()` 兼容，并加 `evo_mode`：

在 `summary` 字典的 `"dataset": dataset,` 之后加 `"evo_mode": run_params["evo_mode"],`。

然后把现有的 `iterations` 推导式（约第81-86行）：

```python
        "iterations": [
            {"iteration": h["iteration"], "accuracy": h["accuracy"],
             "new_patterns": h["new_patterns"], "patterns_total": h["patterns_total"],
             "strategy_version": h["strategy_version"]}
            for h in history
        ],
```

改成（所有字段用 `.get()`，GEPA 独有字段也带上）：

```python
        "iterations": [
            {
                "iteration": h["iteration"],
                "evo_mode": h.get("evo_mode", "classic"),
                "accuracy": h.get("accuracy"),
                "new_patterns": h.get("new_patterns"),
                "patterns_total": h.get("patterns_total"),
                "strategy_version": h.get("strategy_version"),
                # GEPA-only fields (None in classic mode)
                "parent_idx": h.get("parent_idx"),
                "old_minibatch": h.get("old_minibatch"),
                "new_minibatch": h.get("new_minibatch"),
                "new_val_score": h.get("new_val_score"),
                "budget_used": h.get("budget_used"),
            }
            for h in history
        ],
```

- [ ] **Step 5: main() 调用 run_dataset 传新参数**

把 `run_dataset(args.dataset, ...)` 调用改为传 `evo_mode` 和 `train_retrieve_ctx`：

```python
    elif args.dataset:
        run_dataset(args.dataset, args.n_train, args.n_val, args.n_iters,
                    args.seed, args.dynamic_learning, args.output_dir,
                    args.evo_mode, args.train_retrieve_ctx)
```

- [ ] **Step 6: 编译检查全部文件**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; python -m py_compile sro/config.py sro/llm.py sro/engine.py sro/datasets.py sro/grading.py sro/knowledge.py sro/pareto.py sro/__init__.py main.py && echo "=== all compile OK ==="`
Expected: `=== all compile OK ===`

---

### Task 9: .env.example 补示例

**Files:**
- Modify: `.env.example`

**Interfaces:** 无

- [ ] **Step 1: 读 .env.example 现状**

读 `.env.example`，找到 `DYNAMIC_LEARNING=true` 行。

- [ ] **Step 2: 在 DYNAMIC_LEARNING 之后加 GEPA 参数块**

```env
# 测试时不匹配是否走动态学习（true 开 / false 直接用长期策略硬答）
DYNAMIC_LEARNING=true

# ---- 进化模式与训练期检索 ----
# 进化模式：classic（n_iters 轮全量跑） / gepa（Pareto+minibatch+预算控制）
EVO_MODE=classic
# 训练期是否从 KB 检索短期规律作 few-shot 上下文（第89行自递归上下文开关）
TRAIN_RETRIEVE_CTX=true
# GEPA 专属参数（仅 EVO_MODE=gepa 时生效）
MAX_METRIC_CALLS=150
MINIBATCH_SIZE=8
MAX_PROMPT_LENGTH=2000
```

- [ ] **Step 3: 最终全量编译 + 冒烟**

Run: `cd "d:/研究生/claude code相关/Strategy-Rule-Optimization" && find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; python -m py_compile sro/*.py main.py && echo "=== final compile OK ==="`
Expected: `=== final compile OK ===`

然后（用户同意后）跑：`python main.py --demo` 验证 classic 模式不崩。
