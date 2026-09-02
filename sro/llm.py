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

    # ---- API kwargs 构建（移植自 gepa_aime_v2.build_api_kwargs）----
    @staticmethod
    def _sdk_major() -> int:
        try:
            import openai
            return int(getattr(openai, "__version__", "0").split(".")[0])
        except Exception:
            return 0

    @staticmethod
    def _inject(kwargs: dict, key: str, value) -> None:
        """非标准参数注入 extra_body（SDK v1+）或 kwargs（v0）。"""
        if _OpenAIClient._sdk_major() >= 1:
            kwargs.setdefault("extra_body", {})[key] = value
        else:
            kwargs[key] = value

    def build_api_kwargs(
        self, model: str, *, temperature: float, max_tokens: int,
        enable_thinking: bool, max_context_len: int, extra_params: str,
    ) -> dict:
        """构建 chat.completions.create 的 kwargs（含 thinking / context 注入）。

        逻辑同 gepa_aime_v2.build_api_kwargs：
          - extra_params(JSON) 白名单字段直填，其余进 extra_body
          - thinking 注入按模型族：qwen/qwq→enable_thinking；glm→reasoning；
            其他→reasoning_effort；未被 extra_params 显式覆盖时才注入
          - max_context_len>0 → extra_body.max_model_len
        """
        import json as _json
        kwargs: dict = {"model": model, "temperature": temperature,
                        "max_tokens": max_tokens}
        # extra_params 解析（全局共享）
        if extra_params and extra_params.strip():
            try:
                override = _json.loads(extra_params)
                for k, v in override.items():
                    if k in ("model", "temperature", "max_tokens", "tools",
                             "tool_choice", "reasoning_effort", "stream"):
                        kwargs[k] = v
                    else:
                        self._inject(kwargs, k, v)
            except _json.JSONDecodeError as e:
                print(f"  [WARNING] Invalid EXTRA_PARAMS JSON: {e}")
        # thinking 注入（若 extra_params 未显式覆盖）
        _has = ("reasoning_effort" in kwargs
                or "enable_thinking" in kwargs
                or "enable_thinking" in kwargs.get("extra_body", {})
                or "reasoning" in kwargs.get("extra_body", {}))
        ml = (model or "").lower()
        if not _has:
            if enable_thinking:
                if "qwen" in ml or "qwq" in ml:
                    self._inject(kwargs, "enable_thinking", True)
                elif "glm" in ml:
                    self._inject(kwargs, "reasoning", True)
                else:
                    kwargs["reasoning_effort"] = "medium"
            else:
                # qwen3（非 qwen3.）默认开思考，非流式必须关；glm 同理关 reasoning
                if "qwen3" in ml and "qwen3." not in ml:
                    self._inject(kwargs, "enable_thinking", False)
                elif "glm" in ml:
                    self._inject(kwargs, "reasoning", False)
        # 上下文长度
        if max_context_len > 0 and "max_model_len" not in kwargs.get("extra_body", {}):
            self._inject(kwargs, "max_model_len", max_context_len)
        return kwargs

    @staticmethod
    def _needs_stream(kwargs: dict) -> bool:
        """需要走流式的情形（对齐 gepa_aime_v3._chat_complete）。

        - extra_body 里存在 enable_thinking 键（不论 True/False）：
          qwen3 即使设 False 也得走流式，否则端点报 400
          "enable_thinking only support stream call"
        - extra_body 里存在 reasoning 键（GLM 同理）
        - 显式 stream=True
        """
        eb = kwargs.get("extra_body") or {}
        return ("enable_thinking" in eb
                or "reasoning" in eb
                or kwargs.get("stream") is True)

    @staticmethod
    def _aggregate_stream(stream) -> str:
        """聚合流式 delta：取 content，为空则回退 reasoning_content。

        用 getattr 防 usage/role 等无 choices 字段的 chunk 抛异常
        （对齐 gepa_aime_v3 的健壮写法）。
        """
        parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            c = getattr(delta, "content", None)
            if c:
                parts.append(c)
            r = (getattr(delta, "reasoning_content", None)
                 or getattr(delta, "reasoning", None))
            if r:
                reasoning_parts.append(r)
        text = "".join(parts)
        if text.strip():
            return text
        # 某些模型把全部输出放在 reasoning_content，回退保证有东西可抽答案
        return "".join(reasoning_parts)

    def chat(self, model: str, system: str, user: str, *,
             timeout: Optional[float] = None,
             temperature: float = 0.0, max_tokens: int = 4096,
             enable_thinking: bool = False, max_context_len: int = 0,
             extra_params: str = "") -> str:
        """返回模型文本输出。is_real=False 时回退占位。

        开启 thinking 的模型族走流式（端点硬性要求），聚合 delta.content；
        否则非流式（现状路径不变）。
        """
        if not self.is_real:
            return f"[TaskLM placeholder reply] system={system[:40]!r} user={user[:40]!r}"
        kwargs = self.build_api_kwargs(
            model, temperature=temperature, max_tokens=max_tokens,
            enable_thinking=enable_thinking, max_context_len=max_context_len,
            extra_params=extra_params,
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if self._needs_stream(kwargs):
            kwargs["stream"] = True
            stream = self._client.chat.completions.create(
                messages=messages, timeout=timeout, **kwargs)
            return self._aggregate_stream(stream)
        resp = self._client.chat.completions.create(
            messages=messages, timeout=timeout, **kwargs)
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
# Embedder —— 向量化（本地 sentence-transformers，不依赖中转端点）
# ---------------------------------------------------------------------------


class Embedder:
    """文本向量化。

    默认用本地 sentence-transformers（all-MiniLM-L6-v2，dim=384），
    离线、免费、语义检索够用。首次运行下载 ~80MB 模型到 HF 缓存。
    未装 sentence-transformers 时回退哈希伪向量（仅作演示，无语义）。
    """

    def __init__(self, model: Optional[str] = None, dim: Optional[int] = None) -> None:
        cfg = get_config()
        self.model = model or cfg.embed_model
        self.dim = dim or cfg.embed_dim
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model)
            # 取真实维度（新版本改名 get_embedding_dimension，做兼容）
            get_dim = getattr(self._st_model, "get_embedding_dimension",
                              getattr(self._st_model, "get_sentence_embedding_dimension", None))
            if get_dim is not None:
                self.dim = get_dim()
        except Exception:
            self._st_model = None  # 回退哈希伪向量

    def embed(self, text: str) -> list[float]:
        """返回文本向量。有本地模型用真实 embedding，否则哈希伪向量。"""
        if self._st_model is not None:
            return self._st_model.encode(text).tolist()
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
        self.temperature = cfg.task_temperature
        self.max_tokens = cfg.task_max_tokens
        self.enable_thinking = cfg.task_enable_thinking
        self.max_context_len = cfg.task_max_context_len
        self.extra_params = cfg.extra_params
        self.embedder = embedder or Embedder()
        # 判分回调：由 engine 注入对应数据集的 evaluate_answer。
        # 为 None 时回退到内置 is_correct（numeric/exact/freeform 简单判分）。
        self.judger: Optional[callable] = None

    def _call_llm(self, system: str, user: str) -> str:
        """真实模型调用。无 key 时由 _OpenAIClient 自动回退占位。"""
        return _get_client().chat(
            self.model, system, user, timeout=self.timeout,
            temperature=self.temperature, max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
            max_context_len=self.max_context_len,
            extra_params=self.extra_params,
        )

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

        覆盖模型常见输出格式（按优先级）：
          1. \\boxed{...}        —— LaTeX 标准答案包装
          2. "The answer is X" / "最终答案: X" / "答案是X"
          3. <answer>X</answer> 标签
          4. $X$ / \\$X$ / **X** 行内包装里的数字
          5. 文本中最后一个数字（兜底）
        找不到则返回去掉首尾空白的原文。
        """
        import re
        # 1) \boxed{...}（用最后一个 \boxed，通常是最终答案）
        last = raw.rfind("\\boxed")
        if last != -1:
            start = raw.find("{", last)
            if start != -1:
                depth = 0
                for i in range(start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return raw[start + 1:i].strip()
                return raw[start + 1:].strip()
        # 2) "The answer is X" / "最终答案: X" / "答案是X"
        m = re.search(
            r"(?:final answer|the answer is|answer is|最终答案|答案(?:是为|是|:))\s*[:：]?\s*(.+?)(?:[.。]?\s*$|\n)",
            raw, re.IGNORECASE,
        )
        if m:
            cand = m.group(1).strip()
            # 若候选是 "X words" 这类，提取首个数字
            return cand
        # 3) <answer>X</answer> 标签
        m = re.search(r"<answer>\s*(.+?)\s*</answer>", raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # 4) $X$ / \$X$ / **X** 行内包装
        m = re.search(r"\$\\?([-\d,.]+)\$|\*\*([-\d,.]+)\*\*", raw)
        if m:
            return next(g for g in m.groups() if g)
        # 5) 文本中最后一个数字（含逗号千分位/小数/分数）
        #    先匹配带逗号的千分位（整体），再匹配套路纯数字
        nums = re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:/\d+)?", raw)
        if nums:
            return nums[-1].replace(" ", "")
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
                new_text = self._call_llm(sys, usr)
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
        self.temperature = cfg.reflection_temperature
        self.max_tokens = cfg.reflection_max_tokens
        self.enable_thinking = cfg.reflection_enable_thinking
        self.max_context_len = cfg.reflection_max_context_len
        self.extra_params = cfg.extra_params
        self.embedder = embedder or Embedder()

    def _call_llm(self, system: str, user: str) -> str:
        """真实模型调用。无 key 时回退占位。"""
        return _get_client().chat(
            self.model, system, user, timeout=self.timeout,
            temperature=self.temperature, max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
            max_context_len=self.max_context_len,
            extra_params=self.extra_params,
        )

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
                text = self._call_llm(sys, usr)
                p = Example(text=text, source_run_id=t.run_id, polarity=-1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            for t in correct[:3]:
                sys = "You are an experience distillation assistant. Extract one reusable effective practice from this correct trajectory (one sentence)."
                usr = f"Problem:\n{t.problem}\nTrajectory excerpt:\n{t.trajectory[:200]}"
                text = self._call_llm(sys, usr)
                p = Example(text=text, source_run_id=t.run_id, polarity=+1)
                p.embedding = self.embedder.embed(text)
                patterns.append(p)
            # 长期策略
            sys = "You are a meta-learning methodology expert. Summarize a general problem-solving strategy."
            usr = f"Summarize the common lessons from these {len(traces)} trajectories."
            long_text = self._call_llm(sys, usr)
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
