"""机制 1：双证据条款晋升门控（报告 §4.6）。

条款从规则层晋升到策略层，当且仅当：
    1. 分类器判定为"可泛化原则"（而非"狭窄边缘案例补丁"）
    2. recurrence >= m_rec（跨任务簇复现）
    3. support >= k_promote（正向证据计数）

局部证据：本批反思反馈（覆盖本批哪些失败、覆盖几个）
全局证据：规则库历史复现频率（对应 TextReg 的 RuleBank）

分类不确定时默认留在规则层（保守侧）——宁可策略层学得慢，
不可被狭窄条款污染。

分类器提示模板：src/prompts/templates/principle_classifier.jinja2
"""
