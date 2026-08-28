"""策略层：条款（Clause）数据结构。

对应技术路线报告 §4.6——每条条款带独立的泛用性/有效性记账：

    Clause = {
        text:           条款文本
        origin:         来源（由哪批样本的反思产生）
        scope:          泛用性分 s̄（语义分析器测量，§4.6 机制 2）
        recurrence:     历史复现频率（客观计数，与 scope 交叉验证）
        support:        正向证据计数（机制 1 晋升门控用）
        blame:          负向归因计数
        verified_delta: 条款级 A/B 测得的准确率增量（机制 4）
        inefficiency:   ℐᵢ = |cᵢ|_tok · (1 − s̄ᵢ)（机制 2）
    }

StrategyMemory 维护条款集合与 Σℐ 漂移监控（每代上报，防"逐步变窄"）。
"""
