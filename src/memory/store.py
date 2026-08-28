"""三层记忆的统一存储。

MemoryStore 负责：
    - 加载/保存 strategy.json / rules.json / support.json
    - 临时规则隔离（§4.5）：provisional 规则单独存放，任务结束时可选择丢弃或
      批量过晋升门控（§4.6 机制 1-3）后才沉淀为长期记忆
    - 训练期与测试期的读写分离：长期记忆只在有显式验证的训练期更新
"""
