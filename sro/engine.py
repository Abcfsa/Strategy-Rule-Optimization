"""SROEngine —— 编排两阶段闭环。

阶段一 train_and_reflect：TaskLM→反思→产出短期规律+长期策略→
                       用长期策略迭代 TaskLM prompt（候选→打分→筛选）→ 闭环。
阶段二 inference：测试题→向量检索短期规律→命中/不匹配两分支→回答。
"""

from __future__ import annotations

from typing import Optional

from .llm import (
    Embedder,
    Example,
    ReflectionLM,
    Strategy,
    TaskLM,
    Trace,
    TrainSample,
)
from .knowledge import KnowledgeBase


class SROEngine:
    """系统编排器，持有两个 LM 与一个共享知识库。"""

    def __init__(
        self,
        task_lm: Optional[TaskLM] = None,
        reflection_lm: Optional[ReflectionLM] = None,
        embedder: Optional[Embedder] = None,
        kb: Optional[KnowledgeBase] = None,
        match_threshold: float = 0.6,   # 命中阈值
        top_k: int = 3,                   # 检索条数
    ) -> None:
        self.embedder = embedder or Embedder()
        self.task_lm = task_lm or TaskLM(self.embedder)
        self.reflection_lm = reflection_lm or ReflectionLM(self.embedder)
        self.kb = kb or KnowledgeBase(self.embedder)
        self.match_threshold = match_threshold
        self.top_k = top_k

    def set_dataset(self, dataset: str) -> None:
        """绑定数据集：把对应 evaluate_answer 注入 TaskLM.judger。

        dataset: gsm8k / math / aime / hotpotqa。判分逻辑复用
        openai_api_test 中已验证的函数，保证与基线一致。
        """
        from .datasets import _import_eval
        judger, _ = _import_eval(dataset)
        self.task_lm.judger = judger

    # ===================================================================
    # 阶段一：训练与反思迭代
    # ===================================================================

    def train_and_reflect(
        self,
        train_set: list[TrainSample],
        n_iters: int = 3,
        candidates_per_iter: int = 2,
        verbose: bool = True,
    ) -> None:
        """训练闭环。

        train_set 元素为 TrainSample（含 problem + 标准答案），反思据此区分对/错轨迹。

        每轮：
          1. TaskLM 跑训练集 → 轨迹+结果（用标准答案判对错）
          2. ReflectionLM 反思 → 短期规律 + 长期策略
          3. 写入知识库
          4. 用长期策略迭代 TaskLM 的 prompt：生成候选策略→
             在训练子集上打分→筛选保留最优→更新 TaskLM
          5. 回到 1，共 n_iters 轮
        """
        for it in range(1, n_iters + 1):
            if verbose:
                print(f"\n=== Train iteration {it}/{n_iters} ===")

            # 1) TaskLM 运行训练集（带当前策略 + 当前已有短期规律）
            traces: list[Trace] = []
            for sample in train_set:
                # 训练期也可检索已积累的短期规律作为上下文
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

            # 2) 反思
            short_patterns, long_strategy = self.reflection_lm.reflect(traces)

            # 3) 写入知识库
            self.kb.add_patterns(short_patterns)
            # 长期策略先暂存，下面用候选打分决定是否更新

            # 4) 策略迭代：生成候选 → 打分 → 筛选
            best_strategy = self._evolve_strategy(
                long_strategy, traces, train_set, candidates_per_iter
            )
            self.kb.update_strategy(best_strategy)
            self.task_lm.update_strategy(best_strategy)

            if verbose:
                acc = sum(t.result.correct for t in traces) / len(traces)
                print(f"  Accuracy this round: {acc:.2%}")
                print(f"  New patterns: {len(short_patterns)}"
                      f" (total {len(self.kb.examples)})")
                print(f"  Strategy version: v{best_strategy.version}")

    def _evolve_strategy(
        self,
        reflected_strategy: Strategy,
        traces: list[Trace],
        train_set: list[TrainSample],
        n_candidates: int,
    ) -> Strategy:
        """策略迭代核心：候选生成 → 在训练子集上打分 → 保留最优。

        真实实现可在此引入 GEPA 式 Pareto 前沿筛选；本阶段用简单最高分。
        """
        # 候选 = 反思产出的策略 + TaskLM 基于反馈的若干变体
        candidates: list[Strategy] = [reflected_strategy]
        feedback = f"Accuracy this round: {sum(t.result.correct for t in traces)}/{len(traces)}"
        candidates += self.task_lm.mutate(feedback, candidates=n_candidates)

        # 在训练子集上打分每个候选（临时换上候选策略跑一个子集）
        best, best_score = reflected_strategy, -1.0
        original_strategy = self.task_lm.strategy
        eval_subset = train_set[: max(1, len(train_set) // 3)]
        for cand in candidates:
            self.task_lm.update_strategy(cand)
            cand_traces = [
                self.task_lm.run(
                    s.problem,
                    gold_answer=s.answer,
                    answer_type=s.answer_type,
                )
                for s in eval_subset
            ]
            cand_score = sum(self.task_lm.score(t) for t in cand_traces) / len(cand_traces)
            cand.score = cand_score
            if cand_score > best_score:
                best, best_score = cand, cand_score

        # 恢复（train_and_reflect 主循环会用 best 统一更新）
        self.task_lm.update_strategy(original_strategy)
        return best

    # ===================================================================
    # 阶段二：测试与推理
    # ===================================================================

    def inference(self, question: str, verbose: bool = False) -> tuple[str, dict]:
        """测试推理，含命中/不匹配两条分支。

        返回 (answer, meta)，meta 记录走了哪条分支、命中了哪些规律。
        """
        # ---- 匹配机制：向量检索短期规律 ----
        hits = self.kb.retrieve(question, k=self.top_k,
                                threshold=self.match_threshold)
        hit = bool(hits)

        meta: dict = {"branch": "match" if hit else "miss",
                      "matched_examples": [e.text[:40] for e in hits]}

        if hit:
            # ===== 命中分支：例子 + 长期策略 → TaskLM =====
            trace = self.task_lm.run(question, context_examples=hits)
            answer = trace.result.answer
            meta["dynamic_added"] = False
            if verbose:
                print(f"[MATCH] hit {len(hits)} patterns -> answering directly")
        else:
            # ===== 不匹配分支：动态学习机制（架构图标注"待考虑"）=====
            answer, trace = self._dynamic_learning(question, meta, verbose)

        # 清理本轮临时规律，避免污染下一题
        self.kb.drop_tentative()
        return answer, meta

    def _dynamic_learning(
        self, question: str, meta: dict, verbose: bool
    ) -> tuple[str, Trace]:
        """不匹配时的动态学习：临时归纳规律→临时入库→第二轮测试。

        架构图中标注"待考虑"的部分：本阶段实现为可选的临时归纳，
        默认不永久入库（drop_tentative 会在 inference 末尾清掉）。
        """
        # 1) ReflectionLM 从该问题临时归纳一条规律
        new_pattern = self.reflection_lm.extract_pattern_from_question(question)

        # 2) 临时入库（permanent=False，不污染长期规律池）
        self.kb.tentative_add(new_pattern)

        # 3) 第二轮测试：用长期策略 + 这条临时规律重新检索并回答
        if verbose:
            print("[MISS] dynamic learning triggered: induce temporary pattern -> second-round attempt")
        hits = self.kb.retrieve(question, k=self.top_k,
                               threshold=0.0)  # 临时放宽，确保取到刚加的
        trace = self.task_lm.run(question, context_examples=hits)

        meta["dynamic_added"] = True
        meta["tentative_pattern"] = new_pattern.text[:40]
        return trace.result.answer, trace
