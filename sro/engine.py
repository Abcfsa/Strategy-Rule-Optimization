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


class SROEngine:
    """系统编排器，持有两个 LM 与一个共享知识库。"""

    def __init__(
        self,
        task_lm: Optional[TaskLM] = None,
        reflection_lm: Optional[ReflectionLM] = None,
        embedder: Optional[Embedder] = None,
        kb: Optional[KnowledgeBase] = None,
        match_threshold: float = 0.6,   # 命中阈值
        top_k: int = 3,                  # 检索条数
        dynamic_learning: bool = True,   # miss 时是否走动态学习
        test_use_patterns: bool = True,  # 测试时是否注入短期规律作 context
        evo_mode: str = "classic",
        train_retrieve_ctx: bool = True,
        max_metric_calls: int = 150,
        minibatch_size: int = 8,
        max_prompt_length: int = 2000,
        seed: int = 42,
        use_merge: bool = True,
        max_merge_invocations: int = 10,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.task_lm = task_lm or TaskLM(self.embedder)
        self.reflection_lm = reflection_lm or ReflectionLM(self.embedder)
        self.kb = kb or KnowledgeBase(self.embedder)
        self.match_threshold = match_threshold
        self.top_k = top_k
        self.dynamic_learning = dynamic_learning
        self.test_use_patterns = test_use_patterns
        self.evo_mode = evo_mode
        self.train_retrieve_ctx = train_retrieve_ctx
        self.max_metric_calls = max_metric_calls
        self.minibatch_size = minibatch_size
        self.max_prompt_length = max_prompt_length
        self.seed = seed
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        # merge 调度状态（对齐 GEPA MergeProposer）
        self._merges_due = 0
        self._total_merges_tested = 0
        self._last_iter_found_new_program = False
        self._merges_performed: list[tuple[int, int, int]] = []

    def set_dataset(self, dataset: str) -> None:
        """绑定数据集：把对应 evaluate_answer 注入 TaskLM.judger，
        并把该数据集的输出格式指令注入 TaskLM.dataset_format。

        dataset: gsm8k / math / aime / hotpotqa。判分逻辑复用
        openai_api_test 中已验证的函数，保证与基线一致。
        """
        from .datasets import _import_eval
        from .llm import DATASET_FORMAT_INSTRUCTIONS
        judger, _ = _import_eval(dataset)
        self.task_lm.judger = judger
        self.task_lm.dataset_format = DATASET_FORMAT_INSTRUCTIONS.get(dataset, "")

    # ===================================================================
    # 阶段一：训练与反思迭代
    # ===================================================================

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

    # -------------------------------------------------------------------
    # GEPA 辅助方法（_train_gepa 内部使用）
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # merge 算子辅助方法（适配自 gepa/proposer/merge.py）
    # -------------------------------------------------------------------

    def _find_common_ancestor(
        self, candidates: list[Strategy], rng, max_attempts: int = 10,
    ) -> Optional[tuple[int, int, int]]:
        """从候选池找 (idx_a, idx_b, ancestor_idx) 三元组用于 merge。

        适配 GEPA find_common_ancestor_pair + filter_ancestors +
        does_triplet_have_desirable_predictors。SRO 单亲树（parent_idx
        单 int），祖先走 while 循环而非递归；无命名组件，"有可合并物"
        的检查退化为文本差异检查。
        """
        n = len(candidates)
        if n < 3:
            return None

        def _ancestors_of(idx: int) -> set:
            """走 parent_idx 链收集祖先集（SRO 单亲树）。"""
            seen: set[int] = set()
            cur = candidates[idx].parent_idx
            while cur is not None and cur not in seen:
                seen.add(cur)
                cur = candidates[cur].parent_idx
            return seen

        for _ in range(max_attempts):
            if n < 2:
                return None
            i, j = rng.sample(range(n), 2)
            if j < i:
                i, j = j, i

            anc_i = _ancestors_of(i)
            anc_j = _ancestors_of(j)
            # 互不为祖先后代
            if j in anc_i or i in anc_j:
                continue

            common = anc_i & anc_j
            valid = []
            for a in common:
                # 跳过已 merge 的三元组
                if (i, j, a) in self._merges_performed:
                    continue
                # 祖先分数不能高于任一后代
                if (candidates[a].score > candidates[i].score
                        or candidates[a].score > candidates[j].score):
                    continue
                # 单文本"有可合并物"检查：至少一方相对祖先进化了
                if (candidates[i].text != candidates[a].text
                        or candidates[j].text != candidates[a].text):
                    valid.append(a)
            if not valid:
                continue
            # 按 ancestor 分数加权采样
            weights = [candidates[a].score + 1e-9 for a in valid]
            ancestor = rng.choices(valid, weights=weights, k=1)[0]
            return (i, j, ancestor)
        return None

    @staticmethod
    def _select_eval_subsample(
        scores_a: list[float], scores_b: list[float], rng,
        num_subsample_ids: int = 5,
    ) -> list[int]:
        """分层采样 val 子集用于 merge 评估。

        忠实移植 GEPA select_eval_subsample_for_merged_program：按
        a 更好 / b 更好 / 平 三组各采 ~1/3，补足从剩余随机。
        """
        import math
        all_indices = set(range(len(scores_a)))
        p1 = [i for i, (s1, s2) in enumerate(zip(scores_a, scores_b))
              if s1 > s2]
        p2 = [i for i, (s1, s2) in enumerate(zip(scores_a, scores_b))
              if s2 > s1]
        p3 = [i for i in all_indices if i not in p1 and i not in p2]

        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(p1), n_each)
        n2 = min(len(p2), n_each)
        n3 = min(len(p3), num_subsample_ids - (n1 + n2))
        selected: list[int] = []
        if n1:
            selected += rng.sample(p1, k=n1)
        if n2:
            selected += rng.sample(p2, k=n2)
        if n3:
            selected += rng.sample(p3, k=n3)

        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += rng.sample(unused, k=remaining)
            else:
                selected += rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]

    def _eval_candidate_on_subsample(
        self, text: str, minibatch: list[TrainSample],
    ) -> list[float]:
        """在子采样上评估策略文本，返回 per-sample 1.0/0.0。

        _eval_candidate 的裁剪版：建临时 Strategy，跑子采样（无 KB
        context，与 _eval_candidate 一致），恢复。
        """
        from .llm import Strategy as _S
        original = self.task_lm.strategy
        self.task_lm.update_strategy(_S(text=text, version=-1))
        scores: list[float] = []
        for sample in minibatch:
            trace = self.task_lm.run(
                sample.problem,
                context_examples=[],
                gold_answer=sample.answer,
                answer_type=sample.answer_type,
            )
            scores.append(1.0 if trace.result.correct else 0.0)
        self.task_lm.update_strategy(original)
        return scores

    def _attempt_merge(
        self, candidates: list[Strategy], val_for_pareto: list[TrainSample],
        n_val: int, budget_used: int, rng, verbose: bool, iteration: int,
    ) -> tuple[Optional[Strategy], int, dict]:
        """尝试一次 merge（GEPA MergeProposer.propose 内联）。

        返回 (新策略或None, 更新后的budget, 记录dict)。
        调用方负责 append 到 candidates + 重算 Pareto。
        """
        triplet = self._find_common_ancestor(candidates, rng)
        if triplet is None:
            return (None, budget_used, {"attempted": False, "reason": "no_triplet"})

        idx_a, idx_b, anc_idx = triplet
        self._merges_performed.append((idx_a, idx_b, anc_idx))
        self._total_merges_tested += 1

        parent_a = candidates[idx_a]
        parent_b = candidates[idx_b]
        ancestor = candidates[anc_idx]

        # 子采样选择（基于两父本的 val_scores）
        subsample_ids = self._select_eval_subsample(
            parent_a.val_scores, parent_b.val_scores, rng)
        minibatch = [val_for_pareto[k] for k in subsample_ids]

        # 父本各自在子采样上的分数（从 val_scores 取，无需重跑）
        parent_a_sub = [parent_a.val_scores[k] for k in subsample_ids]
        parent_b_sub = [parent_b.val_scores[k] for k in subsample_ids]
        parent_a_sum = sum(parent_a_sub)
        parent_b_sum = sum(parent_b_sub)

        # 父本跑子采样取 traces（为 merge_strategies 提供诊断）
        traces_a = self._run_minibatch(parent_a, minibatch)
        traces_b = self._run_minibatch(parent_b, minibatch)
        budget_used += 2 * len(subsample_ids)

        # LLM 融合
        merged_text = self.reflection_lm.merge_strategies(
            ancestor, parent_a, parent_b, traces_a, traces_b, iteration)
        merged_text = _clean_markdown(merged_text)
        if not merged_text or merged_text == parent_a.text or merged_text == parent_b.text:
            return (None, budget_used, {
                "attempted": True, "accepted": False,
                "reason": "empty_or_identical",
                "merged_entities": (idx_a, idx_b, anc_idx),
                "subsample_ids": subsample_ids,
                "parent_a_subsample_sum": parent_a_sum,
                "parent_b_subsample_sum": parent_b_sum,
            })
        if len(merged_text) > self.max_prompt_length:
            return (None, budget_used, {
                "attempted": True, "accepted": False,
                "reason": f"too_long({len(merged_text)})",
                "merged_entities": (idx_a, idx_b, anc_idx),
                "subsample_ids": subsample_ids,
                "parent_a_subsample_sum": parent_a_sum,
                "parent_b_subsample_sum": parent_b_sum,
            })

        # 子采样评估融合产物
        merged_sub_scores = self._eval_candidate_on_subsample(
            merged_text, minibatch)
        budget_used += len(subsample_ids)
        merged_sum = sum(merged_sub_scores)

        record = {
            "attempted": True, "accepted": False,
            "merged_entities": (idx_a, idx_b, anc_idx),
            "subsample_ids": subsample_ids,
            "parent_a_subsample_scores": parent_a_sub,
            "parent_b_subsample_scores": parent_b_sub,
            "merged_subsample_scores": merged_sub_scores,
            "parent_a_subsample_sum": parent_a_sum,
            "parent_b_subsample_sum": parent_b_sum,
            "merged_subsample_sum": merged_sum,
            "threshold": max(parent_a_sum, parent_b_sum),
            "merge_total_tested": self._total_merges_tested,
        }

        # 接受准则：>= max(parent_sums)（非 strict，忠实 GEPA）
        if merged_sum < max(parent_a_sum, parent_b_sum):
            record["reason"] = (f"no_improvement({merged_sum}"
                                f" < {max(parent_a_sum, parent_b_sum)})")
            if verbose:
                print(f"[Merge-Reject] subsample {merged_sum}"
                      f" < max({parent_a_sum},{parent_b_sum})")
            return (None, budget_used, record)

        # 接受：全 val eval
        merged_strat = Strategy(
            text=merged_text, version=len(candidates), parent_idx=idx_a)
        merged_strat.val_scores = self._eval_candidate(
            merged_strat, val_for_pareto)
        budget_used += n_val
        merged_strat.score = sum(merged_strat.val_scores) / n_val
        # sidecar：第二父本 + 祖先（不改 dataclass）
        merged_strat._merge_meta = {"parent_b_idx": idx_b, "ancestor_idx": anc_idx}

        record["accepted"] = True
        if verbose:
            print(f"[Merge-Accept] #{len(candidates)} via"
                  f" ({idx_a},{idx_b},anc={anc_idx})"
                  f" sub={merged_sum} val={merged_strat.score:.2%}")
        return (merged_strat, budget_used, record)

    # -------------------------------------------------------------------
    # GEPA 核心训练循环
    # -------------------------------------------------------------------

    def _train_gepa(self, train_set: list[TrainSample],
                    verbose: bool) -> list[dict]:
        """GEPA-style training: Pareto front + minibatch + budget + keep-better.

        Mirrors gepa_aime_v3.py main loop. KB coexists: short patterns from
        reflection are added to KB alongside the Pareto candidate pool.
        """
        import random
        from .pareto import build_pareto_fronts, select_candidate_from_pareto_front

        if verbose:
            print("[Info] GEPA mode is budget-controlled (max_metric_calls); n_iters is ignored.")

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
                print(f"\n=== GEPA iteration {iteration}"
                      f" (budget {budget_used}/{self.max_metric_calls}) ===")

            # ===== Merge attempt (before reflective, 对齐 GEPA) =====
            merge_record = None
            if (self.use_merge and self._merges_due > 0
                    and self._last_iter_found_new_program
                    and self._total_merges_tested < self.max_merge_invocations
                    and budget_used < self.max_metric_calls):
                merged_strat, budget_used, merge_record = self._attempt_merge(
                    candidates, val_for_pareto, n_val,
                    budget_used, rng, verbose, iteration)
                if merged_strat is not None:
                    # 接受：消费 slot，入池，重算 Pareto，跳过 reflective
                    self._merges_due = max(0, self._merges_due - 1)
                    candidates.append(merged_strat)
                    pareto_fronts = build_pareto_fronts(
                        [c.val_scores for c in candidates], n_val)
                    if merged_strat.score > best_score:
                        best_idx = len(candidates) - 1
                        best_score = merged_strat.score
                    merge_record.update({
                        "iteration": iteration, "evo_mode": "gepa-merge",
                        "budget_used": budget_used,
                        "new_val_score": merged_strat.score,
                        "strategy_version": merged_strat.version,
                        "strategy_text": merged_strat.text,
                    })
                    history.append(merge_record)
                    self._last_iter_found_new_program = True
                    continue
                else:
                    # 拒绝：不消费 merges_due（对齐 GEPA），落到 reflective
                    pass
            self._last_iter_found_new_program = False  # reset；reflective 接受时再设 True

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
                    print(f"[Eval] Parent minibatch:"
                          f" {old_sum}/{len(minibatch)}"
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
                        print(f"[Reject] No improvement"
                              f" ({new_sum} <= {old_sum}).")
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
                    print(f"[Accept] Candidate #{len(candidates) - 1}"
                          f" val={new_strat.score:.2%}")
                if new_strat.score > best_score:
                    best_idx = len(candidates) - 1
                    best_score = new_strat.score
                    if verbose:
                        print(f"  * New best! ({best_score:.2%})")

                # 2h) short patterns from reflection -> KB (coexistence)
                patterns, _ = self.reflection_lm.reflect(
                    old_traces + new_traces)
                self.kb.add_patterns(patterns)

                # 调度下一轮 merge（对齐 GEPA schedule_if_needed）
                if (self.use_merge
                        and self._total_merges_tested < self.max_merge_invocations):
                    self._merges_due += 1
                self._last_iter_found_new_program = True

                history.append({
                    "iteration": iteration, "evo_mode": "gepa",
                    "parent_idx": parent_idx,
                    "old_minibatch": old_sum, "new_minibatch": new_sum,
                    "new_val_score": new_strat.score,
                    "accepted": True, "budget_used": budget_used,
                    "strategy_version": new_strat.version,
                    "strategy_text": new_strat.text,
                    "merge_attempted_this_iter": merge_record is not None,
                    "merge_record": merge_record,
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
                  f" val={best_score:.2%}, budget"
                  f" {budget_used}/{self.max_metric_calls}")
        return history

    # ===================================================================
    # 阶段二：测试与推理
    # ===================================================================

    def inference(self, question: str, verbose: bool = False) -> tuple[str, dict]:
        """测试推理，含命中/不匹配两条分支。

        返回 (answer, meta)，meta 记录走了哪条分支、命中了哪些规律。
        miss 时：若 dynamic_learning 开启则走动态学习，否则直接硬答。
        """
        # ---- 匹配机制：向量检索短期规律 ----
        if self.test_use_patterns:
            hits = self.kb.retrieve(question, k=self.top_k,
                                    threshold=self.match_threshold)
        else:
            hits = []
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
        elif self.dynamic_learning:
            # ===== 不匹配分支：动态学习机制 =====
            answer, trace = self._dynamic_learning(question, meta, verbose)
        else:
            # ===== 不匹配且关闭动态学习：直接用长期策略硬答 =====
            trace = self.task_lm.run(question, context_examples=[])
            answer = trace.result.answer
            meta["dynamic_added"] = False
            if verbose:
                print("[MISS] dynamic learning OFF -> answering directly with strategy only")

        # 清理本轮临时规律，避免污染下一题
        self.kb.drop_tentative()
        meta["raw"] = trace.trajectory
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
        if self.test_use_patterns:
            hits = self.kb.retrieve(question, k=self.top_k,
                                   threshold=0.0)  # 临时放宽，确保取到刚加的
        else:
            hits = []  # test_use_patterns 关闭：规律照归纳入 KB，但不注入 context
        trace = self.task_lm.run(question, context_examples=hits)

        meta["dynamic_added"] = True
        meta["tentative_pattern"] = new_pattern.text[:40]
        return trace.result.answer, trace
