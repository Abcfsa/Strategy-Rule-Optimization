"""支持集：SupportEntry 数据结构。

对应技术路线报告 §4.2 支持集层与 §4.7.3（AdaMEM 长短期混合、DC snippet）：

    SupportEntry = {
        problem_digest:  问题摘要
        strategy_tags:   使用的策略标注（命中哪些条款）
        key_steps:       关键推理步骤摘要（AdaMEM short-term 形态）
        full_trace:      完整轨迹（可选，AdaMEM long-term 形态，预算允许时注入）
        final_answer:    最终答案
        verified:        是否验证过正确（Voyager 纪律：只收正确轨迹）
    }

SupportMemory 维护支持集与检索接口（§4.3：语义粗筛 → 置信度加权 → LLM 精选）。
注入数量上限 = 3（MemAPO Top-K=3 峰值结论）。
"""
