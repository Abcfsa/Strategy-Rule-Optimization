"""知识库：存储短期规律（可检索）与长期策略。

对应系统架构：
    - 短期规律 (Example)  → 支持向量相似度检索，用于阶段二匹配
    - 长期策略 (Strategy) → 唯一最新版本，作为 TaskLM 的 system prompt
    - 临时入库机制        → 阶段二"不匹配"分支的动态学习
"""

from __future__ import annotations

import math
from typing import Optional

from .llm import Embedder, Example, Strategy


class KnowledgeBase:
    """统一存储短期规律 + 长期策略。

    线性扫描检索（数据量小阶段足够）；真实场景可换 FAISS / Chroma。
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or Embedder()
        self.examples: list[Example] = []   # 短期规律池
        self.strategy: Strategy = Strategy(text="", version=0)  # 长期策略

    # ---------------- 短期规律 ----------------

    def add_pattern(self, example: Example) -> None:
        """新增短期规律。若缺 embedding 则补齐。"""
        if example.embedding is None:
            example.embedding = self.embedder.embed(example.text)
        self.examples.append(example)

    def add_patterns(self, examples: list[Example]) -> None:
        for e in examples:
            self.add_pattern(e)

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.6) -> list[Example]:
        """向量相似度检索 top-k 短期规律。

        返回相似度 >= threshold 的前 k 条；不足则返回更少（交给调用方
        据此进入"不匹配"分支）。
        """
        if not self.examples:
            return []
        q_vec = self.embedder.embed(query)
        scored = [
            (self._cosine(q_vec, e.embedding), e)
            for e in self.examples
            if e.embedding is not None
        ]
        scored = [(s, e) for s, e in scored if s >= threshold]
        scored.sort(key=lambda se: se[0], reverse=True)
        return [e for _, e in scored[:k]]

    def best_similarity(self, query: str) -> float:
        """查询与规律池的最高相似度，供"是否命中"判定。"""
        if not self.examples:
            return 0.0
        q_vec = self.embedder.embed(query)
        return max(
            self._cosine(q_vec, e.embedding)
            for e in self.examples
            if e.embedding is not None
        )

    # ---------------- 临时入库（动态学习）----------------

    def tentative_add(self, example: Example) -> None:
        """临时加入一条规律（permanent=False），用于阶段二第二轮测试。

        默认不污染长期规律池：阶段二结束时可按 permanent 决定是否转正。
        """
        example.permanent = False
        self.add_pattern(example)

    def promote_tentative(self, run_id: str) -> None:
        """把某次动态学习的临时规律转正为永久规律。"""
        for e in self.examples:
            if not e.permanent and e.source_run_id == run_id:
                e.permanent = True

    def drop_tentative(self) -> None:
        """丢弃所有临时规律（不保留到下一题）。"""
        self.examples = [e for e in self.examples if e.permanent]

    # ---------------- 长期策略 ----------------

    def update_strategy(self, strategy: Strategy) -> None:
        """更新长期策略（仅保留最新版本，旧版被覆盖）。"""
        self.strategy = strategy

    # ---------------- 持久化占位 ----------------

    def save(self, path: str) -> None:
        """保存到磁盘。占位：真实实现用 JSON / pickle 序列化 examples 与 strategy。"""
        raise NotImplementedError("Persistence will be added in a later stage")

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        raise NotImplementedError("Persistence will be added in a later stage")

    # ---------------- 工具 ----------------

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
