"""提示模板渲染。

render(template_name, **context)：
    加载 templates/<template_name>.jinja2 并用 jinja2 渲染。

模板清单（与技术路线报告 §4.6/§4.7 对应）：
    reflect_mutate.jinja2          反射变异（条款 diff + 归因 + 规则候选）
    principle_classifier.jinja2    §4.6 机制 1 原则/补丁分类器
    rule_induction.jinja2          对比规则归纳（ETGPO/AIR）
    judge.jinja2                   LLM Evolution Judge 预筛
    retrieval_scorer.jinja2        §4.3 检索精选打分
    adapt_rule_extract.jinja2      DG 式规则提取
    feedback.jinja2                评测反馈构造（失败/成功 ~1:1）
"""
