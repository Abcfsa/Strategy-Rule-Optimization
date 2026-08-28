"""LLM Evolution Judge 预筛（Toolbox，报告 §4.7.2）。

在条款候选生成后、条款级 A/B（昂贵验证）之前：
    - 独立 LLM 调用判定候选质量（同模型 + greedy decoding）
    - 判为 "bad" 则要求重新生成，最多 JUDGE_MAX_RETRIES 次
    - 节省条款级 A/B 的验证调用预算

提示模板：src/prompts/templates/judge.jinja2
"""
