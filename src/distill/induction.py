"""对比规则归纳（报告 §4.7.2，AIR 思想）。

在每个错误类别内构造平衡 A/B 示例集：
    - 作对的样本（同簇内成功案例）
    - 做错的样本（该错误类别的失败案例）
配对后让归纳器输出结构化规则：

    rule = {
        trigger:  "适用条件（if...）",
        action:   "推荐动作 / 应避免的错误（then...）",
        polarity: positive | negative,
        evidence: [正确示例, 错误示例],   # 详细 + 带示例（ETGPO 消融要求）
    }

归纳提示模板：src/prompts/templates/rule_induction.jinja2
产出送入 src/regularize.dual_evidence 走晋升门控（默认先入规则层）。
"""
