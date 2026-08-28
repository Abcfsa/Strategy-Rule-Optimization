"""ETGPO 式错误分类（报告 §4.7.2）。

四步 Top-Down 流程的前两步：
    1. 错误收集：失败轨迹（含完整推理，不截断）
    2. 错误分类：优化器 LLM 分析失败轨迹并分配错误类别
       每个类别 = {名称, 摘要, 描述, 示例, 错误类型, 解释}
    3. 类别筛选：过滤只关联 1 个问题的类别，按失败数排序取 top

启发式错误分类（廉价信号，配合 LLM 分类）：
    format_error / insufficient_reasoning / calculation_error / conceptual_error
    （复用 gepa_aime_v2.py 的 _classify_error_type 思路）
"""
