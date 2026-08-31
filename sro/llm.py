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


@dataclass
class TrainSample:
    """训练样本：题目 + 标准答案。

    answer_type:
        exact    —— 字符串归一化后精确匹配（默认）
        numeric  —— 数值比对（容差 1e-6），支持整数/小数/分数 a/b
        freeform —— 自由文本，占位返回 False（后续接 LLM judge）
    """

    problem: str
    answer: str
    answer_type: str = "exact"


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
            return f"[TaskLM placeholder reply] system={system[:40]!r} user={user[:40]!r}"
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
        # 判分回调：由 engine 注入对应数据集的 evaluate_answer。
        # 为 None 时回退到内置 is_correct（numeric/exact/freeform 简单判分）。
        self.judger: Optional[callable] = None

    def _call_llm(self, system: str, user: str) -> str:
        """真实模型调用。无 key 时由 _OpenAIClient 自动回退占位。"""
        return _get_client().chat(self.model, system, user, timeout=self.timeout)

    # ---- 核心运行 ----
    def run(
        self,
        problem: str,
        context_examples: Optional[list[Example]] = None,
        gold_answer: Optional[str] = None,
        answer_type: str = "exact",
    ) -> Trace:
        """在策略 + 检索到的短期规律下执行题目。

        context_examples: 命中的短期规律（few-shot 上下文）。
        gold_answer:      标准答案；提供则据 answer_type 判对错。
        answer_type:      exact / numeric / freeform。
        """
        few_shot = "\n".join(e.to_prompt_str() for e in (context_examples or []))
        system_prompt = self._assemble_system_prompt(few_shot)
        # 记录附加上下文，供反思分析
        ctx = {"strategy_version": self.strategy.version,
               "n_examples": len(context_examples or []),
               "examples_text": few_shot,
               "answer_type": answer_type}
        raw = self._call_llm(system_prompt, problem)
        answer, correct = self._parse_output(raw, gold_answer, answer_type)
        return Trace(problem=problem, trajectory=raw,
                     result=Result(answer=answer, correct=correct), context=ctx)

    def _assemble_system_prompt(self, few_shot: str) -> str:
        parts = [self.strategy.text or "You are a careful problem solver."]
        if few_shot:
            parts.append("Relevant past experiences:\n" + few_shot)
        return "\n\n".join(parts)

    # ---- 答案抽取 + 判分 ----
    @staticmethod
    def extract_answer(raw: str) -> str:
        """从 LLM 原始输出抽取最终答案。

        优先级：\\boxed{...} > "Final answer:"/"最终答案:" 后内容 >
        末尾数字。找不到则返回去掉首尾空白的原文。
        """
        import re
        # 1) \boxed{...}（含嵌套花括号的贪婪匹配）
        m = re.search(r"\\boxed\{(.+?)\}", raw)
        if m:
            return m.group(1).strip()
        # 2) "Final answer:" / "最终答案:" / "答案是:" 等
        m = re.search(
            r"(?:final answer|最终答案|答案(?:是为|是|:)|answer is|the answer is)\s*[:：]?\s*(.+)$",
            raw, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().splitlines()[-1].strip()
        # 3) 末尾独立数字
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", raw.strip())
        if m:
            return m.group(1)
        return raw.strip()

    @staticmethod
    def normalize_answer(ans: str, answer_type: str) -> str | float | None:
        """归一化答案。numeric 返回 float，exact 返回 str，失败 None。"""
        import re
        s = (ans or "").strip()
        # 先剥 \boxed{...} 取其内容（再处理残留的裸 \ 与 {}）
        m = re.search(r"\\boxed\{(.+?)\}", s)
        if m:
            s = m.group(1)
        s = re.sub(r"[\\{}]", "", s)      # 去残留反斜杠 / 花括号
        s = s.replace(" ", "")
        if answer_type == "numeric":
            try:
                if "/" in s:             # 分数 a/b
                    a, b = s.split("/")
                    return float(a) / float(b)
                return float(s)
            except (ValueError, ZeroDivisionError):
                return None
        return s.lower()                  # exact：小写化字符串

    @classmethod
    def is_correct(cls, predicted: str, gold: str, answer_type: str) -> bool:
        """内置简单判分（无 judger 时回退用）。

        predicted 可以是 \boxed{...} 形式（normalize 会剥离）。
        numeric 用「绝对 1e-6 或相对 1e-4」双容差，覆盖 2/3 vs 0.6667
        这类四舍五入。
        """
        if answer_type == "freeform":
            return False                 # freeform 必须靠 judger（SQuAD 等）
        p = cls.normalize_answer(predicted, answer_type)
        g = cls.normalize_answer(gold, answer_type)
        if p is None or g is None:
            return False
        if answer_type == "numeric":
            if p == g:
                return True
            denom = max(abs(g), 1e-12)
            return abs(p - g) < 1e-6 or abs(p - g) / denom < 1e-4
        return p == g                    # exact 字符串比对

    def _parse_output(self, raw: str, gold_answer: Optional[str],
                      answer_type: str) -> tuple[str, bool]:
        """抽取答案并据 gold_answer 判对错。无 gold_answer 时默认错。

        优先用注入的 judger（数据集专用判分）；否则回退内置 is_correct。
        """
        answer = self.extract_answer(raw)
        if gold_answer is None:
            return answer, False
        if self.judger is not None:
            try:
                return answer, bool(self.judger(answer, gold_answer))
            except Exception:
                return answer, False
        return answer, self.is_correct(answer, gold_answer, answer_type)

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
                sys = ("You are a prompt optimizer. Rewrite the current strategy "
                       "based on the following feedback. Keep it concise and "
                       "reusable. Output only the rewritten strategy text.")
                usr = (f"Current strategy:\n{self.strategy.text}\n\n"
                       f"Feedback:\n{feedback}\n\nOutput rewrite variant #{i+1}.")
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
                sys = "You are an experience distillation assistant. Extract one reusable pitfall-avoidance lesson from this failed trajectory (one sentence)."
                usr = f"Problem:\n{t.problem}\nWrong answer:\n{t.result.answer}\nTrajectory excerpt:\n{t.trajectory[:200]}"
                text = c.chat(self.model, sys, usr, timeout=self.timeout)
                p = Example(text=text, source_run_id=t.run_id, polarity=-1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            for t in correct[:3]:
                sys = "You are an experience distillation assistant. Extract one reusable effective practice from this correct trajectory (one sentence)."
                usr = f"Problem:\n{t.problem}\nTrajectory excerpt:\n{t.trajectory[:200]}"
                text = c.chat(self.model, sys, usr, timeout=self.timeout)
                p = Example(text=text, source_run_id=t.run_id, polarity=+1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            # 长期策略
            sys = "You are a meta-learning methodology expert. Summarize a general problem-solving strategy."
            usr = f"Summarize the common lessons from these {len(traces)} trajectories."
            long_text = c.chat(self.model, sys, usr, timeout=self.timeout)
        else:
            # 占位逻辑（无 key）
            for t in wrong[:3]:
                p = Example(
                    text=f"Common mistake on problems like '{t.problem[:20]}': {t.result.answer[:30]}",
                    source_run_id=t.run_id, polarity=-1,
                )
                p.embedding = self.embedder.embed(p.text)
                patterns.append(p)
            for t in correct[:3]:
                p = Example(
                    text=f"Effective practice: {t.trajectory[:60]}",
                    source_run_id=t.run_id, polarity=+1,
                )
                p.embedding = self.embedder.embed(p.text)
                patterns.append(p)
            long_text = self._call_llm(
                system="You are a meta-learning methodology expert. Summarize a general problem-solving strategy.",
                user=f"Summarize the common lessons from these {len(traces)} trajectories.",
            )

        long_strategy = Strategy(text=long_text, version=1)
        return patterns, long_strategy

    # ---- 阶段二：单题动态归纳 ----
    def extract_pattern_from_question(self, question: str) -> Example:
        """对未命中的测试问题临时归纳一条短期规律（动态学习机制）。"""
        text = self._call_llm(
            system="Distill one reusable problem-solving takeaway from this single problem.",
            user=question,
        )
        ex = Example(text=text, polarity=+1, permanent=False)
        ex.embedding = self.embedder.embed(text)
        return ex
