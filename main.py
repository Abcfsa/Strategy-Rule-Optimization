"""SRO 演示入口。

用占位 LM 跑通两阶段闭环，验证数据流与分支逻辑。
接入真实模型时，替换 sro/llm.py 中的 _call_llm 占位即可。
"""

from __future__ import annotations

import argparse

from sro import SROEngine


def demo() -> None:
    engine = SROEngine(match_threshold=0.5, top_k=3)

    train_set = [
        "求方程 2x+3=7 的解。",
        "一个三角形三个内角之比为 1:2:3，求最大角。",
        "计算 1+2+...+100 的和。",
        "化简分数 12/18。",
    ]
    test_set = [
        "求方程 5x-2=13 的解。",        # 与训练题相似
        "证明根号2是无理数。",            # 与训练题不相似 → 触发动态学习
    ]

    print("########## 阶段一：训练与反思迭代 ##########")
    engine.train_and_reflect(train_set, n_iters=2, verbose=True)

    print("\n########## 阶段二：测试与推理 ##########")
    for q in test_set:
        print(f"\n问题: {q}")
        answer, meta = engine.inference(q, verbose=True)
        print(f"分支: {meta['branch']} | 动态学习: {meta.get('dynamic_added')}")
        print(f"回答: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRO 两阶段反思式自进化框架（占位演示）",
    )
    parser.add_argument(
        "--demo", action="store_true", help="运行内置演示（占位 LM）",
    )
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
