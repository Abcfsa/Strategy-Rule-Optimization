"""双早停信号（报告 §4.7.1-4.7.2）。

1. Parent-based（Toolbox）：滑动窗口内当前候选不超过父代 max + ηₚ 则停；
   fallback 到 Moment-based：窗口 w=10 内分数均值绝对差 < ηₘ=10⁻³。
2. Semantic Early-Stopping（SEM-Stop）：条款编辑后策略层的 embedding
   距离序列收敛（语义不再变化）即判定饱和（先例：省 38% 预算）。

两者叠加，避免"过了最优点还在变异"的过拟合加速阶段（Category1 报告
§4.2 的 Phase C）。
"""
