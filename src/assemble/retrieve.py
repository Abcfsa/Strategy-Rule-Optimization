"""三层检索实现（报告 §4.3）。

retrieve_rules(query, rule_memory, mode)：
    语义粗筛 → 置信度加权 → 错误模式信号（适应模式）→ LLM 精选

retrieve_examples(query, support_memory)：
    语义粗筛 → LLM 精选，上限 3 条

实现注意：
    - embedding 可选依赖（sentence-transformers），缺省时退化为纯 LLM 打分
    - LLM 精选用批处理（一次调用），避免每样本都调（ERL 教训：
      LLM 检索优于 embedding 检索，但要控制开销）
"""
