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
