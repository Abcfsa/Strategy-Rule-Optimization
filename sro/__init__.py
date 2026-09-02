"""SRO (Strategy-Rule-Optimization) — 两阶段反思式自进化框架。

阶段一 train_and_reflect：TaskLM 跑训练集 → ReflectionLM 反思 → 产出
    短期规律(Specific Examples) + 长期策略(Long-term Strategy) →
    用长期策略迭代 TaskLM 的 prompt（候选→打分→筛选）→ 闭环。

阶段二 inference：测试问题 → 向量检索短期规律 →
    命中：例子+长期策略拼上下文 → TaskLM 回答；
    未命中：动态学习（ReflectionLM 临时归纳规律、临时入库）→ 第二轮测试。
"""

from .llm import (
    TaskLM,
    ReflectionLM,
    Embedder,
    Trace,
    Result,
    Example,
    Strategy,
    TrainSample,
)
from .knowledge import KnowledgeBase
from .engine import SROEngine
from .config import Config, get_config
from . import datasets  # noqa: F401  数据集加载（按需 import 数据库）
from .pareto import (
    build_pareto_fronts,
    is_dominated,
    remove_dominated_programs,
    select_candidate_from_pareto_front,
)

__all__ = [
    "TaskLM",
    "ReflectionLM",
    "Embedder",
    "Trace",
    "Result",
    "Example",
    "Strategy",
    "TrainSample",
    "KnowledgeBase",
    "SROEngine",
    "Config",
    "get_config",
    "build_pareto_fronts",
    "is_dominated",
    "remove_dominated_programs",
    "select_candidate_from_pareto_front",
]
