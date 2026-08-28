"""机制 4：条款级验证门控与记账（报告 §4.6，后置裁决）。

条款合并前的条款级 A/B：
    - 验证 batch 上对比 [S] 与 [S + c_new]（或 [S − c_del]）
    - 接受条件：verified_delta > 0 且在跨域混合 batch 上不劣化
    - 与 RAPOA 两阶段接受检验叠加：
        A/B 跑 optimization seeds，跨域不劣化跑 held-out selection seeds
    - 回滚只回滚单条条款，其余条款不受影响（整体粒度回滚的局部化）

记账（喂回机制 1 的全局证据）：
    - 条款被检索注入后，按作答结果累积复现与有效性证据
    - 命中且答对 → +1；命中且答错 → −1（与 src/assemble 的检索加权联动）
"""
