"""快速适应主协议（报告 §4.5）。

输入：策略层 S（冻结）、规则库 R、支持集 E、适应预算 B 个带反馈样本

    1. warm-up：用 [S] 裸策略跑 B 个样本，记录成败与错误分类
    2. 匹配：失败样本错误类型 → 查询 R；命中 → 规则置信度 +1，标记高优先级
    3. 归纳：未命中的失败模式，与同批成功样本对比归纳
       → 临时规则（标注 provisional，走 src/adapt/rule_extraction.py）
    4. 组装：后续测试样本用 [S ; top(R ∪ R_provisional, x) ; top(E, x)]
       （走 src/assemble）
    5. 约束：临时规则不参与跨任务迁移，任务结束即弃
       （或批量过 §4.6 机制 1-3 门控后才沉淀为长期记忆）

适应过程中由 src/adapt/flip_monitor.py 监控规则层 negative flip，
flip ratio > 1 时停止本轮规则归纳。
"""
