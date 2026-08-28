"""sro.distill — 规则蒸馏（进化 → 记忆的通道之一）。

对应技术路线报告 §4.4 与 §4.7.2（ETGPO 式操作规程，必做消融 E10）：
    taxonomy:      ETGPO 式错误分类——失败轨迹先聚类分类，
                   过滤只关联 1 个问题的类别，按失败数取 top
    induction:     对比归纳——同簇内"作对样本 vs 做错样本"配对，
                   让归纳器输出 if-then 规则（AIR 思想）

消融依据（ETGPO）：跳过分类 −1.37、短指导比详细指导 −2.22，
因此蒸馏产出必须是"详细 + 带示例"的规则条目（evidence 字段）。

错误分类信号复用：gepa_aime_v2.py 的 _classify_error_type 思路
（format_error / insufficient_reasoning / calculation_error / conceptual_error）。
"""
