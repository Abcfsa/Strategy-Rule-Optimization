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

    def __init__(self, embedder: Optional[Embedder] = None,
                 dedup_threshold: float = 0.0) -> None:
        self.embedder = embedder or Embedder()
        self.examples: list[Example] = []   # 短期规律池
        self.strategy: Strategy = Strategy(text="", version=0)  # 长期策略
        # >0 时入库去重：与已有规律 cosine >= 阈值则跳过（rich 模式建议 0.92）
        self.dedup_threshold = dedup_threshold

    # ---------------- 短期规律 ----------------

    def add_pattern(self, example: Example) -> bool:
        """新增短期规律。若缺 embedding 则补齐。

        dedup_threshold > 0 时，与已有规律 cosine >= 阈值则跳过入库。
        返回 True=已入库，False=因重复被跳过。
        """
        if example.embedding is None:
            example.embedding = self.embedder.embed(example.text)
        if self.dedup_threshold > 0.0:
            for e in self.examples:
                if (e.embedding is not None
                        and self._cosine(example.embedding, e.embedding)
                        >= self.dedup_threshold):
                    return False
        self.examples.append(example)
        return True

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

    def retrieve_paired(self, query: str, k: int = 3,
                        threshold: float = 0.6) -> list[Example]:
        """按 source_run_id 分组的"配对检索"：选 k 组，组内全部条目一起返回。

        rich 模式下同一 trace 产两条互补规律（wrong: 错误步骤定位 + 避坑教训；
        correct: 有效做法 + 触发规则），id 相同。本方法保证选择以组为单位：
        k=3 即 3 个不同 run_id（最多 3"对"），组内所有条目（通常 2 条）都注入，
        杜绝"只选中错误定位、丢了对应解决办法"的半对状态。

        组得分 = 组内条目与 query 相似度的最大值；组内至少一条 >= threshold
        该组才入候选。run_id 为空串的规律（聚类规律、动态学习临时项若未带 id）
        各自单独成组。返回按组得分降序排列的扁平列表（组内保持池内顺序）。
        """
        if not self.examples:
            return []
        q_vec = self.embedder.embed(query)
        # 分组：保持首次出现的顺序，组内保持池内顺序
        groups: dict[str, list[float]] = {}
        order: list[str] = []
        for e in self.examples:
            if e.embedding is None:
                continue
            s = self._cosine(q_vec, e.embedding)
            key = e.source_run_id or ""   # 空 id 每条自成一组
            if e.source_run_id:
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(s)
            else:
                # 空 id：无配对语义，单独成组（用临时唯一键）
                tkey = f"__solo__{id(e)}"
                groups[tkey] = [s]
                order.append(tkey)
        # 组得分 = 组内最大相似度；阈值过滤在组级（任一条过线即组过线）
        scored_groups = [(max(groups[g]), g) for g in order]
        scored_groups = [(s, g) for s, g in scored_groups if s >= threshold]
        scored_groups.sort(key=lambda sg: sg[0], reverse=True)
        picked: list[Example] = []
        for _, g in scored_groups[:k]:
            if g.startswith("__solo__"):
                # 单例组：直接找回那条
                for e in self.examples:
                    if e.embedding is not None and not e.source_run_id \
                            and f"__solo__{id(e)}" == g:
                        picked.append(e)
                        break
            else:
                picked.extend(e for e in self.examples
                              if e.source_run_id == g and e.embedding is not None)
        return picked

    def retrieve_group_keys(self, query: str, k: int,
                            threshold: float = 0.6) -> list[str]:
        """配对检索的组选择子步：返回 top-k 组的键（run_id；空 id 返回占位键）。

        供 llm/random 匹配做组级抽样/精选：先选组，再用 expand_group 取整组。
        键格式与 retrieve_paired 内部一致（__solo__<id> 为单例组占位键）。
        """
        if not self.examples:
            return []
        q_vec = self.embedder.embed(query)
        groups: dict[str, list[float]] = {}
        order: list[str] = []
        for e in self.examples:
            if e.embedding is None:
                continue
            s = self._cosine(q_vec, e.embedding)
            if e.source_run_id:
                if e.source_run_id not in groups:
                    groups[e.source_run_id] = []
                    order.append(e.source_run_id)
                groups[e.source_run_id].append(s)
            else:
                tkey = f"__solo__{id(e)}"
                groups[tkey] = [s]
                order.append(tkey)
        scored_groups = [(max(groups[g]), g) for g in order]
        scored_groups = [(s, g) for s, g in scored_groups if s >= threshold]
        scored_groups.sort(key=lambda sg: sg[0], reverse=True)
        return [g for _, g in scored_groups[:k]]

    def expand_group(self, key: str) -> list[Example]:
        """取出一个组（run_id）的全部规律；__solo__<id> 键取对应单条。"""
        if key.startswith("__solo__"):
            for e in self.examples:
                if e.embedding is not None and not e.source_run_id \
                        and f"__solo__{id(e)}" == key:
                    return [e]
            return []
        return [e for e in self.examples
                if e.source_run_id == key and e.embedding is not None]

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
