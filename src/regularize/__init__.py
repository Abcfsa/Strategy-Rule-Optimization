"""sro.regularize — 条款级正则化（TextReg 判断机制的深度融合）。

对应技术路线报告 §4.6 的四个机制：
    dual_evidence:  机制 1——双证据条款晋升门控（局部批次证据 + RuleBank 复现频率
                    + "原则/补丁"分类器，晋升阈值 k_promote / m_recurrence）
    inefficiency:   机制 2——条款级表示无效性 ℐᵢ = |cᵢ|_tok · (1 − s̄ᵢ)
                    （入库门控 / 全局修剪 / Σℐ 漂移监控）
    guided_update:  机制 3——正则化引导的条款候选选择
                    （score = minibatch_gain − λ·ℐ，含纯任务回退）
    gating:         机制 4——条款级 A/B 验证门控与记账
                    （verified_delta > 0 且跨域混合不劣化才接受；局部回滚）

职责分工：机制 1-3 前置门控（候选生成与选择之前），机制 4 后置裁决（合并之前）。
判断信号只负责定位候选，实测增量负责裁决（§4.6）。
"""
