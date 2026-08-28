"""进化主循环。

对应技术路线报告 §4.4。在 GEPA 骨架（候选池 + Pareto 选择 + minibatch 预筛）
之上叠加：
    - 预算控制：max_metric_calls + 双早停（src/evolve/early_stop.py）
    - Pareto 维度扩展：准确率 + 混合域保持率 + Σℐ（策略层紧凑度，§4.4）
    - 每轮结束：规则蒸馏 + 支持集入库评估（进化 → 记忆）

实现时可复用现有代码：
    - Pareto 选择/候选池：openai_api_test/gepa_aime.py 的 is_dominated /
      remove_dominated_programs / select_candidate_from_pareto_front
    - APEX 式分层采样（反射反馈优先取"可修复前沿"样本，§4.4 成本控制）
"""
