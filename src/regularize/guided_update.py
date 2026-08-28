"""机制 3：正则化引导的条款候选选择（报告 §4.6，对应 TextReg Regularization-Guided Update）。

反射变异器输出多个候选条款编辑时，按与正则信号的兼容性排序：
    score(candidate) = minibatch_gain − λ · ℐ_candidate

即任务对齐（minibatch 提升）与正则约束（低 ℐ、分类为原则、有复现证据）的权衡。
λ 由配置给定（E8 中扫描）。

回退机制（TextReg fallback）：所有候选正则分都差时，执行纯任务焦点更新，
但只写入规则层（临时规则），不晋升策略层——保证任务对齐主目标不被
正则项完全压制。
"""
