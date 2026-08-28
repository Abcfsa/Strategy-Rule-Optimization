"""prompt 组装（报告 §4.3）。

build_prompt(query, strategy_memory, rules, examples, budget)：
    按 [策略条款 ; 规则 ; 支持集示例] 顺序拼接，应用硬长度预算。

    - 条款顺序可按有效性分排序
    - 超预算时按优先级裁剪：先裁示例，再裁规则，策略层不裁
    - 返回最终 user-facing prompt 与注入清单（用于记账，§4.6 机制 4）
"""
