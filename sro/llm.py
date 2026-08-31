"""LLM 封装与数据结构。

- 有 OPENAI_API_KEY 时，TaskLM / ReflectionLM / Embedder 走真实 OpenAI 兼容调用。
- 无 key 时回退到占位实现，保证框架可独立演示（数据流与分支逻辑仍验证可达）。
接入非 OpenAI 兼容端点：改 _OpenAIClient 中的 client 构造即可。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import get_config


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """TaskLM 单次运行的结果（可被 reflection 分析）。"""

    answer: str                          # TaskLM 给出的答案
    correct: bool = False                # 是否答对（对照 ground truth）
    score: float = 0.0                   # 该结果的打分（用于候选筛选）


@dataclass
class Trace:
    """TaskLM 单次运行的完整轨迹与结果。

    trajectory 包含中间推理步（思维链 / 工具调用 / 子目标），
    是 ReflectionLM 反思的输入。
    """

    problem: str                          # 输入题目
    trajectory: str                       # 运行轨迹（思维链全文）
    result: Result                         # 最终结果
    context: dict[str, Any] = field(default_factory=dict)  # 当次附加上下文
    run_id: str = ""                      # 运行标识，便于追踪

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = hashlib.md5(
                f"{self.problem}::{self.result.answer}".encode()
            ).hexdigest()[:10]


@dataclass
class Example:
    """短期规律 / 特定例子（Few-shot 风格）。可被向量检索。"""

    text: str                             # 规律/例子的自然语言描述（用于 embedding）
    source_run_id: str = ""               # 来源轨迹
    embedding: Optional[list[float]] = None  # 向量（由 Embedder 填充）
    polarity: int = 1                     # +1 正例(do) / -1 反例(don't)
    permanent: bool = True                # 是否永久入库（动态学习临时项为 False）

    def to_prompt_str(self) -> str:
        """拼成给 TaskLM 看的 few-shot 片段。"""
        tag = "DO" if self.polarity > 0 else "DON'T"
        return f"[{tag}] {self.text}"


@dataclass
class Strategy:
    """长期策略（System Prompt 级别的高级方法论）。"""

    text: str                             # 策略正文
    score: float = 0.0                    # 该策略在验证集上的得分
    version: int = 0                      # 迭代版本号
    parent_version: int = 0              # 由哪个版本演化而来


# ---------------------------------------------------------------------------
# OpenAI 兼容客户端（懒加载，无 openai 库或无 key 时回退占位）
# ---------------------------------------------------------------------------


class _OpenAIClient:
    """OpenAI 兼容客户端单例。无 key 或无 openai 库时 is_real=False。"""

    def __init__(self) -> None:
        self._client = None
        self.is_real = False
        cfg = get_config()
        self._cfg = cfg
        if not cfg.has_api_key:
            return
        try:
            from openai import OpenAI  # type: ignore
            from httpx import Timeout
            # httpx.Timeout 需全四参数，否则 ValueError
            self._client = OpenAI(
                base_url=cfg.openai_base_url,
                api_key=cfg.openai_api_key,
                timeout=Timeout(cfg.task_timeout, connect=30.0,
                                read=cfg.task_timeout, write=30.0, pool=10.0),
            )
            self.is_real = True
        except Exception:
            self._client = None
            self.is_real = False

    def chat(self, model: str, system: str, user: str, timeout: Optional[float] = None) -> str:
        """返回模型文本输出。is_real=False 时回退占位。"""
        if not self.is_real:
            return f"[TaskLM 占位回复] system={system[:40]!r} user={user[:40]!r}"
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""

    def embed(self, model: str, text: str) -> list[float]:
        """返回 embedding 向量。is_real=False 时回退占位伪向量。"""
        if not self.is_real:
            return _hash_embed(text, dim=64)
        resp = self._client.embeddings.create(model=model, input=text)
        return list(resp.data[0].embedding)


_client: Optional[_OpenAIClient] = None


def _get_client() -> _OpenAIClient:
    global _client
    if _client is None:
        _client = _OpenAIClient()
    return _client


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """确定性哈希伪向量（无 API key 时的 fallback）。"""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(b - 128) / 128.0 for b in h]
    while len(vec) < dim:
        vec += vec
    return vec[:dim]


# ---------------------------------------------------------------------------
# Embedder —— 向量化
# ---------------------------------------------------------------------------


class Embedder:
    """文本向量化。有 key 走 OpenAI embeddings，无 key 用哈希伪向量。"""

    def __init__(self, model: Optional[str] = None, dim: Optional[int] = None) -> None:
        cfg = get_config()
        self.model = model or cfg.embed_model
        self.dim = dim or cfg.embed_dim

    def embed(self, text: str) -> list[float]:
        c = _get_client()
        if c.is_real:
            return c.embed(self.model, text)
        return _hash_embed(text, dim=64)


# ---------------------------------------------------------------------------
# TaskLM —— 任务模型
# ---------------------------------------------------------------------------


class TaskLM:
    """任务模型封装，持有当前长期策略。

    run()      : 在给定上下文下执行题目，产出轨迹+结果。
    score()    : 对一次轨迹/结果打分（供候选筛选）。
    mutate()   : 基于反馈生成 prompt 变体（用于迭代）。
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        cfg = get_config()
        self.strategy: Strategy = Strategy(text="", version=0)
        self.model = cfg.task_model
        self.timeout = cfg.task_timeout
        self.embedder = embedder or Embedder()

    def _call_llm(self, system: str, user: str) -> str:
        """真实模型调用。无 key 时由 _OpenAIClient 自动回退占位。"""
        return _get_client().chat(self.model, system, user, timeout=self.timeout)

    # ---- 核心运行 ----
    def run(self, problem: str, context_examples: Optional[list[Example]] = None) -> Trace:
        """在策略 + 检索到的短期规律下执行题目。

        context_examples: 命中的短期规律（few-shot 上下文）。
        """
        few_shot = "\n".join(e.to_prompt_str() for e in (context_examples or []))
        system_prompt = self._assemble_system_prompt(few_shot)
        # 记录附加上下文，供反思分析
        ctx = {"strategy_version": self.strategy.version,
               "n_examples": len(context_examples or []),
               "examples_text": few_shot}
        raw = self._call_llm(system_prompt, problem)
        answer, correct = self._parse_output(raw, problem)
        return Trace(problem=problem, trajectory=raw,
                     result=Result(answer=answer, correct=correct), context=ctx)

    def _assemble_system_prompt(self, few_shot: str) -> str:
        parts = [self.strategy.text or "You are a careful problem solver."]
        if few_shot:
            parts.append("Relevant past experiences:\n" + few_shot)
        return "\n\n".join(parts)

    def _parse_output(self, raw: str, problem: str) -> tuple[str, bool]:
        """从原始输出抽取答案 + 判定对错。占位：返回原文并默认错。

        真实场景：解析 \\boxed{} / 数字；对照 ground truth 判对错。
        """
        return raw, False

    # ---- 打分（候选筛选用）----
    def score(self, trace: Trace) -> float:
        """对一次运行打分。占位：答对 1.0，否则按轨迹长度给微小分。"""
        if trace.result.correct:
            return 1.0
        return min(0.3, len(trace.trajectory) / 1000.0)

    # ---- 策略迭代 ----
    def update_strategy(self, new_strategy: Strategy) -> None:
        """用反思产出的长期策略更新当前策略。"""
        self.strategy = new_strategy

    def mutate(self, feedback: str, candidates: int = 1) -> list[Strategy]:
        """基于反馈生成 prompt 策略变体（用于迭代闭环）。

        有 key 时让 LLM 改写当前策略；无 key 时占位拼接。
        """
        c = _get_client()
        out: list[Strategy] = []
        if c.is_real:
            for i in range(candidates):
                sys = ("你是 prompt 优化器。基于以下反馈改写当前策略，"
                       "保持简洁、可复用，输出改写后的策略正文。")
                usr = (f"当前策略:\n{self.strategy.text}\n\n"
                       f"反馈:\n{feedback}\n\n请输出第 {i+1} 个改写变体。")
                new_text = c.chat(self.model, sys, usr, timeout=self.timeout)
                out.append(Strategy(
                    text=new_text, version=self.strategy.version + 1,
                    parent_version=self.strategy.version,
                ))
        else:
            for i in range(candidates):
                new_text = f"{self.strategy.text}\n# reflection note {i}: {feedback[:80]}"
                out.append(Strategy(
                    text=new_text, version=self.strategy.version + 1,
                    parent_version=self.strategy.version,
                ))
        return out


# ---------------------------------------------------------------------------
# ReflectionLM —— 反思模型
# ---------------------------------------------------------------------------


class ReflectionLM:
    """反思模型封装。

    reflect()  : 分析一批轨迹 → 产出 (短期规律, 长期策略)
    extract_pattern_from_question() : 阶段二"动态学习"用——
                 对单个未命中问题临时归纳一条短期规律。
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        cfg = get_config()
        self.model = cfg.reflection_model
        self.timeout = cfg.reflection_timeout
        self.embedder = embedder or Embedder()

    def _call_llm(self, system: str, user: str) -> str:
        """真实模型调用。无 key 时回退占位。"""
        return _get_client().chat(self.model, system, user, timeout=self.timeout)

    # ---- 阶段一：批量反思 ----
    def reflect(self, traces: list[Trace]) -> tuple[list[Example], Strategy]:
        """分析一批轨迹，产出短期规律 + 长期策略。

        返回:
            short_patterns: 特定例子/短期规律（带 embedding）
            long_strategy:  长期策略（System Prompt 级方法论）
        """
        correct = [t for t in traces if t.result.correct]
        wrong = [t for t in traces if not t.result.correct]
        c = _get_client()

        patterns: list[Example] = []
        if c.is_real:
            # 短期规律：让 LLM 从错题/对题中提炼要点
            for t in wrong[:3]:
                sys = "你是经验提炼助手。从一条错误轨迹中提炼一条可复用的避坑要点（一句话）。"
                usr = f"题目:{t.problem}\n错误答案:{t.result.answer}\n轨迹摘录:{t.trajectory[:200]}"
                text = c.chat(self.model, sys, usr, timeout=self.timeout)
                p = Example(text=text, source_run_id=t.run_id, polarity=-1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            for t in correct[:3]:
                sys = "你是经验提炼助手。从一条正确轨迹中提炼一条可复用的有效做法（一句话）。"
                usr = f"题目:{t.problem}\n轨迹摘录:{t.trajectory[:200]}"
                text = c.chat(self.model, sys, usr, timeout=self.timeout)
                p = Example(text=text, source_run_id=t.run_id, polarity=+1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            # 长期策略
            sys = "你是元学习方法论专家，归纳出一份通用的解题策略。"
            usr = f"总结这 {len(traces)} 条轨迹的共性经验。"
            long_text = c.chat(self.model, sys, usr, timeout=self.timeout)
        else:
            # 占位逻辑（无 key）
            for t in wrong[:3]:
                p = Example(
                    text=f"在处理「{t.problem[:20]}」类问题时常犯错误：{t.result.answer[:30]}",
                    source_run_id=t.run_id, polarity=-1,
                )
                p.embedding = self.embedder.embed(p.text)
                patterns.append(p)
            for t in correct[:3]:
                p = Example(
                    text=f"有效做法：{t.trajectory[:60]}",
                    source_run_id=t.run_id, polarity=+1,
                )
                p.embedding = self.embedder.embed(p.text)
                patterns.append(p)
            long_text = self._call_llm(
                system="你是元学习方法论专家，归纳出一份通用的解题策略。",
                user=f"总结这 {len(traces)} 条轨迹的共性经验。",
            )

        long_strategy = Strategy(text=long_text, version=1)
        return patterns, long_strategy

    # ---- 阶段二：单题动态归纳 ----
    def extract_pattern_from_question(self, question: str) -> Example:
        """对未命中的测试问题临时归纳一条短期规律（动态学习机制）。"""
        text = self._call_llm(
            system="从单个问题中提炼一条可复用的解题要点。",
            user=question,
        )
        ex = Example(text=text, polarity=+1, permanent=False)
        ex.embedding = self.embedder.embed(text)
        return ex
