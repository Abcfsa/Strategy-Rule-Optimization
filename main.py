#!/usr/bin/env python3
"""
SRO Main Entry Point — Strategy-Rule-Optimization

将自进化 prompt 优化中的知识分为三层存储与演化（技术路线报告 GEM 方案）：
    1. 策略层（Strategy）：条款化全局策略，慢速进化，双证据门控 + 条款级正则化
    2. 规则层（Rule）：条件→动作规则，对比归纳产生，可检索注入
    3. 支持集（Support set）：验证过的经验轨迹片段，按查询检索少量注入

子命令（规划）：
    python main.py train --dataset aime --config configs/default.yaml
        训练期：GEPA 进化骨架 + 规则蒸馏 + 条款晋升门控（报告 §4.4 / §4.6）

    python main.py adapt --dataset aime25 --mode zero-shot
        测试期零适应迁移：仅策略层组装（报告 §1.4 成功标准 1）

    python main.py adapt --dataset aime25 --mode few-shot -n 10
        测试期快速适应：DG 式规则提取 + flip ratio 停止信号（报告 §4.5 / §4.7）

当前为框架骨架阶段：命令解析与流程编排骨架就绪，各阶段实现待填充。
"""

import argparse
import logging
from pathlib import Path


# ============================================================================
# SRO CONFIGURATION PARAMETERS
# ============================================================================

# --- 模型设置（参照 gepa_aime.py 的默认值）---
WRITER_MODEL = "qwen3.5-flash"          # 任务 agent（解题）
REFLECTION_MODEL = "qwen3.5-plus"       # 反射/优化器（条款编辑、规则蒸馏、分类）
WRITER_TIMEOUT = 200                    # AIME 推理需要较长超时
REFLECTION_TIMEOUT = 60

# --- 训练期：进化骨架（报告 §4.4，GEPA + 第一类论文成果）---
MAX_METRIC_CALLS = 600                  # 总 writer 调用预算
GENERATIONS = 5                         # G：世代数
CHILDREN_PER_GENERATION = 2             # K：每代子代数
MINIBATCH_SIZE = 8
EARLY_STOP_WINDOW = 10                  # Toolbox: moment-based 窗口
EARLY_STOP_ETA_M = 1e-3                 # Toolbox: ηₘ
JUDGE_MAX_RETRIES = 3                   # Toolbox: LLM Judge 预筛重试上限

# --- 训练期：数据筛选（p1，报告 §4.7.2）---
VARIANCE_FILTER_TOPK = 8                # 方差分解筛选保留的高信息量训练样本数

# --- 条款级正则化（TextReg，报告 §4.6）---
K_PROMOTE = 3                           # 机制 1：晋升所需正向证据计数
M_RECURRENCE = 2                        # 机制 1：晋升所需跨任务簇复现数
REGULARIZATION_LAMBDA = 0.5             # 机制 3：score = gain − λ·ℐ
CLAUSE_II_THRESHOLD = None              # 机制 2：条款 ℐᵢ 入库门限（None=待实验标定，E8）
EMBED_DUP_THRESHOLD = 0.85              # 条款语义去重阈值

# --- 三层检索（报告 §4.3）---
RETRIEVAL_TOPK_COARSE = 20              # embedding 粗筛
RETRIEVAL_MAX_RULES = 5                 # 注入规则上限
RETRIEVAL_MAX_EXAMPLES = 3              # 注入支持集上限（MemAPO Top-K=3 峰值）

# --- 测试期快速适应（报告 §4.5 / §4.7.3-4.7.4）---
ADAPT_BUDGET = 20                       # 适应样本数 B
FLIP_RATIO_THRESHOLD = 1.0              # Overthinking: flip ratio 停止阈值

# --- 数据路径（与现有脚本一致）---
AIME_TRAIN_VAL_PATH = "../AIME/AI-MO___aimo-validation-aime/"
AIME_TEST_PATH = "../AIME/MathArena___aime_2025/"
MATH_TRAIN_PATH = "../MATH/train"
MATH_TEST_PATH = "../MATH/test"


# ============================================================================
# SUBCOMMANDS
# ============================================================================

def cmd_train(args):
    """训练期流水线（占位骨架）：

    1. 加载数据：src/dataset/load.py
    2. p1 方差分解筛选训练样本：src/evolve/data_filter.py
    3. 进化主循环：src/evolve/loop.py（GEPA 骨架）
       - 反射变异（条款 diff）：src/evolve/reflect.py
       - Judge 预筛 + 双早停：src/evolve/judge.py / early_stop.py
       - 条款级正则化四机制：src/regularize/{dual_evidence,inefficiency,guided_update,gating}.py
    4. 规则蒸馏（ETGPO 式分类 + 对比归纳）：src/distill/
    5. 保存三层记忆：src/memory/store.py → outputs/<run>/
    """
    raise NotImplementedError("train pipeline not implemented yet (skeleton stage)")


def cmd_adapt(args):
    """测试期流水线（占位骨架）：

    zero-shot 模式：
        组装 = 策略层 + retrieve(规则, x) + retrieve(支持集, x)：src/assemble/
        评测：src/dataset 加载测试集 + 现有判分逻辑

    few-shot 模式（报告 §4.5）：
        1. warm-up：裸策略跑 B 个样本，记录成败与错误分类
        2. 匹配：错误类型 → 查询规则库，命中提升优先级
        3. 归纳：DG 式规则提取（src/adapt/rule_extraction.py）
        4. flip ratio 监控停止（src/adapt/flip_monitor.py）
        5. 组装后续样本 + Refine 收尾（src/memory/store.py）
        6. 临时规则隔离：任务结束即弃，不写入长期记忆
    """
    raise NotImplementedError("adapt pipeline not implemented yet (skeleton stage)")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SRO: Strategy-Rule-Optimization (skeleton)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练期：进化 + 规则蒸馏 + 条款晋升")
    p_train.add_argument("--dataset", choices=["aime", "math"], default="aime")
    p_train.add_argument("--config", type=str, default="configs/default.yaml")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--output-dir", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_adapt = sub.add_parser("adapt", help="测试期：零适应迁移或快速适应")
    p_adapt.add_argument("--dataset", choices=["aime25", "math", "gsm8k"], default="aime25")
    p_adapt.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    p_adapt.add_argument("-n", "--adapt-samples", type=int, default=ADAPT_BUDGET,
                         help="few-shot 模式下的适应样本数（仅 few-shot 有效）")
    p_adapt.add_argument("--strategy-dir", type=str, required=True,
                         help="train 产出的三层记忆目录")
    p_adapt.add_argument("--config", type=str, default="configs/default.yaml")
    p_adapt.add_argument("--seed", type=int, default=42)
    p_adapt.add_argument("--output-dir", type=str, default=None)
    p_adapt.set_defaults(func=cmd_adapt)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
