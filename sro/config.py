"""配置加载：从 .env 读取，缺省用内置默认值。

优先级：环境变量 > .env 文件 > 本文件 DEFAULTS。
密钥只在 OPENAI_API_KEY，不入代码、不入库（见 .gitignore）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# 可选依赖：python-dotenv。无则降级为仅读环境变量。
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


DEFAULTS = {
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_API_KEY": "",
    "TASK_MODEL": "gpt-4o-mini",
    "REFLECTION_MODEL": "gpt-4o-mini",
    "TASK_TIMEOUT": "200",
    "REFLECTION_TIMEOUT": "60",
    # ---- API 调用参数（task / reflection 各一套，移植自 gepa_aime_v2.py）----
    "TASK_TEMPERATURE": "0.0",
    "REFLECTION_TEMPERATURE": "0.7",
    "TASK_MAX_TOKENS": "4096",
    "REFLECTION_MAX_TOKENS": "2048",
    "TASK_ENABLE_THINKING": "false",
    "REFLECTION_ENABLE_THINKING": "false",
    "TASK_MAX_CONTEXT_LEN": "0",
    "REFLECTION_MAX_CONTEXT_LEN": "0",
    "EXTRA_PARAMS": "",
    # ---- 实验参数（数据规模 / 训练轮次 / 开关）----
    "N_TRAIN": "50",
    "N_VAL": "30",
    "N_ITERS": "3",
    "SEED": "42",
    "DYNAMIC_LEARNING": "true",
    "TEST_USE_PATTERNS": "true",
    # ---- GEPA 进化模式参数 ----
    "EVO_MODE": "classic",
    "TRAIN_RETRIEVE_CTX": "true",
    "MAX_METRIC_CALLS": "150",
    "MINIBATCH_SIZE": "8",
    "MAX_PROMPT_LENGTH": "2000",
    # ---- API 重试（服务端临时故障自动重试）----
    "MAX_RETRIES": "3",           # 0=不重试；可重试错误（5xx/连接/限流/超时）的最大重试次数
    "RETRY_BACKOFF_BASE": "2",    # 指数退避基数（秒），实际等待 base^attempt × (1±0.25 抖动)
    # ---- 短期规律生成 ----
    "PATTERN_GEN_MODE": "basic",  # basic=现状(每reflect最多6条) / rich=多角度+聚类(每reflect最多~35条)
    "PATTERN_MAX_TRACES": "8",    # rich 模式：wrong/correct 各取的最大 trace 数
    "PATTERN_DEDUP_THRESHOLD": "0.0",  # >0 时规律入库前与已有规律 cosine>=阈值则跳过(去重)；rich 建议 0.92
    # ---- 测试时匹配方法 ----
    "TEST_MATCH_METHOD": "vector",  # vector=纯cosine(现状) / llm=向量粗召回+LLM精选(retrieve-then-rerank)
    "LLM_MATCH_RECALL_K": "10",     # llm 模式：向量粗召回的候选数（LLM 从中判断哪些真正适用）
    # ---- 检索 / 匹配 ----
    "EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBED_DIM": "384",
    "MATCH_THRESHOLD": "0.6",
    "TOP_K": "3",
    "GSM8K_PATH": "",
    "MATH_PATH": "",
    "HOTPOTQA_PATH": "",
    "AIME_PATH": "",
    "AIME_GEPA_SPLIT": "false",
}


def _parse_bool(s: str) -> bool:
    """宽松布尔解析：1/true/yes/on → True，其余 False。"""
    return str(s).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    openai_base_url: str
    openai_api_key: str
    task_model: str
    reflection_model: str
    task_timeout: float
    reflection_timeout: float
    # API 调用参数（task / reflection 各一套）
    task_temperature: float
    reflection_temperature: float
    task_max_tokens: int
    reflection_max_tokens: int
    task_enable_thinking: bool
    reflection_enable_thinking: bool
    task_max_context_len: int
    reflection_max_context_len: int
    extra_params: str
    # 实验参数
    n_train: int
    n_val: int
    n_iters: int
    seed: int
    dynamic_learning: bool
    test_use_patterns: bool
    evo_mode: str
    train_retrieve_ctx: bool
    max_metric_calls: int
    minibatch_size: int
    max_prompt_length: int
    # API 重试
    max_retries: int
    retry_backoff_base: float
    # 短期规律生成
    pattern_gen_mode: str
    pattern_max_traces: int
    pattern_dedup_threshold: float
    # 测试时匹配方法
    test_match_method: str
    llm_match_recall_k: int
    embed_model: str
    embed_dim: int
    match_threshold: float
    top_k: int
    gsm8k_path: str = ""
    math_path: str = ""
    hotpotqa_path: str = ""
    aime_path: str = ""
    aime_gepa_split: bool = False

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key) and not self.openai_api_key.startswith("sk-your")

    @classmethod
    def load(cls) -> "Config":
        if _HAS_DOTENV:
            load_dotenv()  # 默认从 ./.env 读
        g = lambda k: os.getenv(k, DEFAULTS[k])
        return cls(
            openai_base_url=g("OPENAI_BASE_URL"),
            openai_api_key=g("OPENAI_API_KEY"),
            task_model=g("TASK_MODEL"),
            reflection_model=g("REFLECTION_MODEL"),
            task_timeout=float(g("TASK_TIMEOUT")),
            reflection_timeout=float(g("REFLECTION_TIMEOUT")),
            task_temperature=float(g("TASK_TEMPERATURE")),
            reflection_temperature=float(g("REFLECTION_TEMPERATURE")),
            task_max_tokens=int(g("TASK_MAX_TOKENS")),
            reflection_max_tokens=int(g("REFLECTION_MAX_TOKENS")),
            task_enable_thinking=_parse_bool(g("TASK_ENABLE_THINKING")),
            reflection_enable_thinking=_parse_bool(g("REFLECTION_ENABLE_THINKING")),
            task_max_context_len=int(g("TASK_MAX_CONTEXT_LEN")),
            reflection_max_context_len=int(g("REFLECTION_MAX_CONTEXT_LEN")),
            extra_params=g("EXTRA_PARAMS"),
            n_train=int(g("N_TRAIN")),
            n_val=int(g("N_VAL")),
            n_iters=int(g("N_ITERS")),
            seed=int(g("SEED")),
            dynamic_learning=_parse_bool(g("DYNAMIC_LEARNING")),
            test_use_patterns=_parse_bool(g("TEST_USE_PATTERNS")),
            evo_mode=g("EVO_MODE"),
            train_retrieve_ctx=_parse_bool(g("TRAIN_RETRIEVE_CTX")),
            max_metric_calls=int(g("MAX_METRIC_CALLS")),
            minibatch_size=int(g("MINIBATCH_SIZE")),
            max_prompt_length=int(g("MAX_PROMPT_LENGTH")),
            max_retries=int(g("MAX_RETRIES")),
            retry_backoff_base=float(g("RETRY_BACKOFF_BASE")),
            pattern_gen_mode=g("PATTERN_GEN_MODE"),
            pattern_max_traces=int(g("PATTERN_MAX_TRACES")),
            pattern_dedup_threshold=float(g("PATTERN_DEDUP_THRESHOLD")),
            test_match_method=g("TEST_MATCH_METHOD"),
            llm_match_recall_k=int(g("LLM_MATCH_RECALL_K")),
            embed_model=g("EMBED_MODEL"),
            embed_dim=int(g("EMBED_DIM")),
            match_threshold=float(g("MATCH_THRESHOLD")),
            top_k=int(g("TOP_K")),
            gsm8k_path=g("GSM8K_PATH"),
            math_path=g("MATH_PATH"),
            hotpotqa_path=g("HOTPOTQA_PATH"),
            aime_path=g("AIME_PATH"),
            aime_gepa_split=_parse_bool(g("AIME_GEPA_SPLIT")),
        )


# 单例（懒加载：首次 import 即读 .env）
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.load()
    return _config
