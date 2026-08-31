"""SRO 入口。

- `python main.py --demo`         ：内置占位演示（无需 API key / 数据集）
- `python main.py --dataset NAME` ：加载真实数据集跑两阶段闭环

数据集：gsm8k / math / aime / hotpotqa（需本地数据 + API key）。
"""

from __future__ import annotations

import argparse

from sro import SROEngine, TrainSample, get_config


def demo() -> None:
    """Built-in placeholder demo to validate data flow and branch logic."""
    engine = SROEngine(match_threshold=0.5, top_k=3)

    # Training set: problems + gold answers (reflection distinguishes correct/wrong)
    train_set = [
        TrainSample("Solve the equation 2x+3=7.", "2", "numeric"),
        TrainSample("A triangle has interior angles in ratio 1:2:3. Find the largest angle.", "90", "numeric"),
        TrainSample("Compute 1+2+...+100.", "5050", "numeric"),
        TrainSample("Simplify the fraction 12/18.", "2/3", "numeric"),
    ]
    # Test set (demo only: data-flow illustration, no gold answers required)
    test_set = [
        "Solve the equation 5x-2=13.",      # similar to a training problem
        "Prove that sqrt(2) is irrational.",  # dissimilar -> triggers dynamic learning
    ]

    print("########## Phase 1: Training & Reflection Loop ##########")
    engine.train_and_reflect(train_set, n_iters=2, verbose=True)

    print("\n########## Phase 2: Testing & Inference ##########")
    for q in test_set:
        print(f"\nQuestion: {q}")
        answer, meta = engine.inference(q, verbose=True)
        print(f"Branch: {meta['branch']} | dynamic_learning: {meta.get('dynamic_added')}")
        print(f"Answer: {answer}")


def run_dataset(dataset: str, n_train: int, n_val: int, n_iters: int,
                seed: int) -> None:
    """Load a real dataset and run the two-phase loop.

    Phase 1: reflect-and-iterate on train; Phase 2: inference + eval on val.
    """
    from sro.datasets import load

    cfg = get_config()
    if not cfg.has_api_key:
        print("[WARN] OPENAI_API_KEY not set; using placeholder LM (data-flow demo only).")

    print(f"########## Loading dataset: {dataset} ##########")
    train, val = load(dataset, n_train=n_train, n_val=n_val, seed=seed)
    print(f"  train: {len(train)} samples | val: {len(val)} samples")

    engine = SROEngine(match_threshold=cfg.match_threshold, top_k=cfg.top_k)
    engine.set_dataset(dataset)   # inject the matching grader

    print(f"\n########## Phase 1: Training & Reflection Loop ({n_iters} iters) ##########")
    engine.train_and_reflect(train, n_iters=n_iters, verbose=True)

    print(f"\n########## Phase 2: Inference eval on val ({len(val)} samples) ##########")
    correct = 0
    for i, sample in enumerate(val, 1):
        answer, meta = engine.inference(sample.problem, verbose=False)
        # grade with the injected judger (consistent with training)
        ok = engine.task_lm.judger(answer, sample.answer) if engine.task_lm.judger else False
        correct += ok
        tag = "OK" if ok else "X"
        print(f"  [{i}/{len(val)}] {tag} | branch={meta['branch']}"
              f" | pred={answer[:40]!r} | gold={sample.answer[:40]!r}")
    print(f"\nval accuracy: {correct}/{len(val)} = {correct / len(val):.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRO: two-phase reflective self-evolution framework",
    )
    parser.add_argument("--demo", action="store_true", help="run built-in placeholder demo")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "aime", "hotpotqa"],
                        help="load a real dataset and run the two-phase loop")
    parser.add_argument("--n-train", type=int, default=50, help="number of train samples (default 50)")
    parser.add_argument("--n-val", type=int, default=30, help="number of val samples (default 30)")
    parser.add_argument("--n-iters", type=int, default=3, help="number of training iterations (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset:
        run_dataset(args.dataset, args.n_train, args.n_val, args.n_iters, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
