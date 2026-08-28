"""sro.adapt — 测试期快速适应。

对应技术路线报告 §4.5 与 §4.7.3-4.7.4：
    protocol:         快速适应主协议（5 步：warm-up → 匹配 → 归纳 → 组装 → 隔离）
    rule_extraction:  Dynamics Grounding 式规则提取（探索→提取→清洗→注入）
    flip_monitor:     flip ratio 停止信号（防适应过程本身过拟合）

关键防过拟合设计（§4.5）：
    - 适应样本与评估样本严格分离
    - 临时规则（provisional）只在本次任务有效，任务结束即弃
    - 长期记忆的更新只发生在有显式验证的训练期
"""
