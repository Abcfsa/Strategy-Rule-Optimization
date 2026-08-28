"""sro.memory — 三层记忆的数据结构与存储。

对应技术路线报告 §4.2：
    策略层（Clause）：条款化全局策略，带 TextReg 式记账字段（§4.6）
    规则层（Rule）：条件→动作规则，带复现计数与极性
    支持集（SupportEntry）：验证过正确的经验轨迹片段

存储：每层一个 JSON 文件（strategy.json / rules.json / support.json），
由 store.py 统一读写。
"""

# 骨架阶段：子模块仅含职责 docstring，实现后启用以下导入
# from .clause import Clause, StrategyMemory
# from .rule import Rule, RuleMemory
# from .support import SupportEntry, SupportMemory
# from .store import MemoryStore

# __all__ = [
#     "Clause", "StrategyMemory",
#     "Rule", "RuleMemory",
#     "SupportEntry", "SupportMemory",
#     "MemoryStore",
# ]
