"""规则层：Rule 数据结构。

对应技术路线报告 §4.2 规则层与 §4.7.3（SCOPE dual-stream、ERL 结构）：

    Rule = {
        trigger:     适用条件描述（检索键）
        action:      推荐动作 / 应避免的错误
        polarity:    positive（怎么做）| negative（别怎么做）
        evidence:    支持案例列表（ETGPO 式：详细指导 + 示例）
        confidence:  验证通过次数 / 使用次数（检索加权，§4.3）
        recurrence:  复现计数（跨批次/任务簇，§4.6 机制 1 全局证据）
        provisional: 是否临时规则（快适应产物，任务结束即弃，§4.5）
    }

RuleMemory 维护规则集合与更新算子：
    - ETGPO 式蒸馏入库（过滤单例类别、按失败数取 top，§4.7.2）
    - SCOPE 式冲突消解与 subsumption 剪枝（§4.7.3）
    - ReMem 式 Refine：剪枝低置信、合并同触发条件（§4.7.4）
"""
