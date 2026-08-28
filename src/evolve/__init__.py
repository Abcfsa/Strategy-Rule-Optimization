"""sro.evolve — 训练期进化骨架（GEPA + 第一类论文成果）。

对应技术路线报告 §4.4 与 §4.7.1-4.7.2：
    loop:        进化主循环（GEPA 骨架：Pareto 选择 + 反射变异 + 预算控制）
    reflect:     反射变异（产出条款 diff + 归因标签 + 规则候选，一次调用多产出）
    judge:       LLM Evolution Judge 预筛（Toolbox：bad 则重新生成，≤3 次）
    early_stop:  双早停信号（Toolbox parent-based + SEM-Stop embedding 收敛）
    data_filter: p1 方差分解数据筛选（进化前预处理）

记忆 ↔ 进化的双向闭环（§4.4）：
    - 记忆 → 进化：反射变异时看到相关规则，避免把特定规则硬塞进策略层
    - 进化 → 记忆：每轮结束跑规则蒸馏（src/distill）与支持集入库评估
"""
