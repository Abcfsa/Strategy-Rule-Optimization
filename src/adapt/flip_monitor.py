"""flip ratio 停止信号（报告 §4.7.3，arXiv:2604.10739）。

监控"新规则是否把原本答对的题改错"（规则层的 negative flip）：
    positive flip：错误 → 正确（规则修复了问题）
    negative flip：正确 → 错误（规则把对题改错）
    flip ratio = negative / positive

当 flip ratio > FLIP_RATIO_THRESHOLD（默认 1.0）时停止本轮规则归纳——
防止适应过程本身过拟合。

初始阈值参照：Overthinking 论文在 AIME 上的 ~7K tokens 交叉点数据。
配合 Self-Correction Help 的 Verify-First（§4.7.1）：只有确认当前作答
错误才应用规则修正，从源头减少 negative flip。
"""
