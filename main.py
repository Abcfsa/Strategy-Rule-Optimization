"""SRO entry point.

- `python main.py --demo`          : built-in placeholder demo (no API key / dataset)
- `python main.py --dataset NAME`  : load a real dataset, run two-phase loop, save outputs

Datasets: gsm8k / math / math500 / aime / hotpotqa / ifbench / livebench / pupa
(need local data + API key; ifbench/livebench additionally need the verifier
deps in requirements-if.txt).

CLI args override .env values; .env overrides built-in defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

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


def _save_outputs(
    out_dir: Path, dataset: str, cfg, run_params: dict,
    history: list[dict], val_results: list[dict], engine: SROEngine,
    initial_baseline: list[dict] | None = None,
    no_pattern_rerun: list[dict] | None = None,
) -> None:
    """Save run artifacts to out_dir (mirrors gepa_aime_v3 multi-file output).

    run_params: the actual values used this run (CLI overrides applied),
    so config.json/summary.json reflect what executed, not the .env defaults.
    no_pattern_rerun: Phase 2b 对照组逐题结果（TEST_NO_PATTERN_RERUN 开启时）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — full config snapshot via asdict (auto-captures all fields),
    # minus the API key (never write secrets to disk).
    config_snapshot = dataclasses.asdict(cfg)
    config_snapshot.pop("openai_api_key", None)
    config_snapshot["dataset"] = dataset
    config_snapshot.update(run_params)
    (out_dir / "config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.json — high-level training + final-eval results (uses actual run params)
    # 注意：val_results 现为最终评估集（test，GEPA 协议）的逐题结果；
    # val_correct/val_total/val_accuracy 字段名保留以兼容旧分析脚本，
    # 实际语义由 eval_on 字段标明（"test" 或回退 "val"）。
    val_correct = sum(1 for r in val_results if r["correct"])
    # val_results/baseline 不落盘明细（轨迹大）；基线只进 summary 汇总
    summary = {
        "dataset": dataset,
        "evo_mode": run_params["evo_mode"],
        "n_train": run_params["n_train"],
        "n_val": run_params["n_val"],
        "n_test": run_params["n_test"],
        "eval_on": run_params["eval_on"],
        "n_iters": run_params["n_iters"],
        "seed": run_params["seed"],
        "dynamic_learning": run_params["dynamic_learning"],
        "final_strategy_version": history[-1]["strategy_version"] if history else 0,
        "total_patterns": len(engine.kb.examples),
        "iterations": [
            {
                "iteration": h["iteration"],
                "evo_mode": h.get("evo_mode", "classic"),
                "accuracy": h.get("accuracy"),
                "new_patterns": h.get("new_patterns"),
                "patterns_total": h.get("patterns_total"),
                "strategy_version": h.get("strategy_version"),
                # GEPA-only fields (None in classic mode)
                "parent_idx": h.get("parent_idx"),
                "old_minibatch": h.get("old_minibatch"),
                "new_minibatch": h.get("new_minibatch"),
                "new_val_score": h.get("new_val_score"),
                "budget_used": h.get("budget_used"),
            }
            for h in history
        ],
        "val_correct": val_correct,
        "val_total": len(val_results),
        "val_accuracy": (val_correct / len(val_results) if val_results else 0.0),
    }
    # 初始 prompt 基线（Phase 0）：训练前在同一评估集（test）上的成绩，
    # 与 Phase 2 最终成绩直接可比；提升量 = 最终 − 初始。
    if initial_baseline:
        base_correct = sum(1 for r in initial_baseline if r["correct"])
        summary["initial_prompt"] = {
            "correct": base_correct,
            "total": len(initial_baseline),
            "accuracy": (base_correct / len(initial_baseline)
                         if initial_baseline else 0.0),
        }
        if val_results:
            summary["val_improvement_vs_initial"] = (
                val_correct / len(val_results) - base_correct / len(initial_baseline))
    else:
        summary["initial_prompt"] = None
        summary["val_improvement_vs_initial"] = None
    # Phase 2b 对照组（TEST_NO_PATTERN_RERUN）：同 strategy 关 pattern 复测，
    # 与正常 Phase 2 之差即 pattern 注入的净贡献
    if no_pattern_rerun:
        np_correct = sum(1 for r in no_pattern_rerun if r["correct"])
        summary["no_pattern_rerun"] = {
            "correct": np_correct,
            "total": len(no_pattern_rerun),
            "accuracy": (np_correct / len(no_pattern_rerun)
                         if no_pattern_rerun else 0.0),
        }
        if val_results:
            summary["pattern_injection_gain"] = (
                val_correct / len(val_results) - np_correct / len(no_pattern_rerun))
    else:
        summary["no_pattern_rerun"] = None
        summary["pattern_injection_gain"] = None
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # strategy.txt — final long-term strategy full text
    (out_dir / "strategy.txt").write_text(
        engine.task_lm.strategy.text, encoding="utf-8")

    # patterns.json — all short-term patterns in the knowledge base
    patterns_out = [
        {"text": e.text, "polarity": e.polarity,
         "permanent": e.permanent, "source_run_id": e.source_run_id}
        for e in engine.kb.examples
    ]
    (out_dir / "patterns.json").write_text(
        json.dumps(patterns_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # val_results.json — per-question pred/gold/branch/correct + trajectory
    (out_dir / "val_results.json").write_text(
        json.dumps(val_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # val_no_pattern.json — Phase 2b 对照组逐题成绩（no-pattern rerun 开启时）
    if no_pattern_rerun is not None:
        (out_dir / "val_no_pattern.json").write_text(
            json.dumps(no_pattern_rerun, ensure_ascii=False, indent=2),
            encoding="utf-8")

    # history.json — per-iteration training records (incl. strategy snapshots)
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nOutputs saved to: {out_dir}")
    print("  - config.json        (run configuration snapshot)")
    print("  - summary.json       (training + val summary)")
    print("  - strategy.txt       (final long-term strategy)")
    print("  - patterns.json      (all short-term patterns)")
    print("  - val_results.json   (per-question results + trajectories)")
    print("  - history.json       (per-iteration training records)")
    if no_pattern_rerun is not None:
        print("  - val_no_pattern.json (no-pattern rerun per-question results)")


def run_dataset(
    dataset: str, n_train, n_val, n_test, n_iters: int,
    seed: int, dynamic_learning: bool, output_dir: str | None,
    evo_mode: str, train_retrieve_ctx: bool, test_use_patterns: bool,
    gepa_split: bool = False, pattern_gen_mode: str | None = None,
    test_match_method: str | None = None, reflect_wrong_only: bool | None = None,
    regularized_verify: bool | None = None, npo_window: int | None = None,
    cross_aware_gen: bool | None = None,
    initial_prompt: str | None = None,
    test_no_pattern_rerun: bool | None = None,
    extract_patterns_on_reject: bool | None = None,
    task_temperature: float | None = None,
    reflection_temperature: float | None = None,
    task_max_tokens: int | None = None,
    reflection_max_tokens: int | None = None,
    task_thinking_budget: int | None = None,
    reflection_thinking_budget: int | None = None,
    skip_initial_baseline: bool = False,
) -> None:
    """Load a real dataset and run the two-phase loop, then save outputs.

    GEPA 协议三集分工：Phase 0 (optional) 初始 prompt 基线跑 test；
    Phase 1 在 train 上反思迭代，候选接受与否由 val 表现决定（gepa/npo 模式）；
    Phase 2 最终推理评估跑 test（优化结束后只评一次）。test 为空时 Phase 0/2
    回退用 val 并告警。Phase 2b (optional) 同 strategy 关 pattern 重测 test，
    隔离 pattern 注入的净贡献。
    """
    from sro.datasets import load

    cfg = get_config()
    if not cfg.has_api_key:
        print("[WARN] OPENAI_API_KEY not set; using placeholder LM (data-flow demo only).")

    print(f"########## Loading dataset: {dataset} ##########")
    # 单数据集配额必须是非负整数；混合数据集允许 int 或 "a,b" 字符串
    # （parse_mixed_count 在 load 内校验）
    if "+" not in dataset:
        try:
            n_train, n_val, n_test = int(n_train), int(n_val), int(n_test)
        except (TypeError, ValueError):
            raise ValueError(
                f"single dataset '{dataset}' requires integer n_train/n_val/n_test; "
                f"got {n_train!r}/{n_val!r}/{n_test!r} "
                f"(comma quotas like '50,30' are only valid for mixed 'a+b' datasets)")
        if min(n_train, n_val, n_test) < 0:
            raise ValueError("n_train/n_val/n_test must be non-negative")
    train, val, test = load(dataset, n_train=n_train, n_val=n_val,
                            n_test=n_test, seed=seed, gepa_split=gepa_split)
    print(f"  train: {len(train)} samples | val: {len(val)} samples"
          f" | test: {len(test)} samples")
    if not test:
        print("  [WARN] test set empty; Phase 0/2 fall back to val"
              " (NOT GEPA-protocol comparable).")
    eval_set = test if test else val
    eval_name = "test" if test else "val (fallback: empty test)"

    engine = SROEngine(
        match_threshold=cfg.match_threshold, top_k=cfg.top_k,
        dynamic_learning=dynamic_learning,
        test_use_patterns=test_use_patterns,
        evo_mode=evo_mode, train_retrieve_ctx=train_retrieve_ctx,
        max_metric_calls=cfg.max_metric_calls, minibatch_size=cfg.minibatch_size,
        max_prompt_length=cfg.max_prompt_length, seed=seed,
        pattern_gen_mode=pattern_gen_mode,
        test_match_method=test_match_method,
        reflect_wrong_only=reflect_wrong_only,
        regularized_verify=regularized_verify,
        npo_window=npo_window,
        cross_aware_gen=cross_aware_gen,
        initial_prompt=initial_prompt,
        extract_patterns_on_reject=extract_patterns_on_reject,
        task_temperature=task_temperature,
        reflection_temperature=reflection_temperature,
        task_max_tokens=task_max_tokens,
        reflection_max_tokens=reflection_max_tokens,
        task_thinking_budget=task_thinking_budget,
        reflection_thinking_budget=reflection_thinking_budget,
    )
    engine.set_dataset(dataset, gepa_split=gepa_split)   # inject the matching grader

    # Phase 0：初始 prompt 基线（策略=初始版本、KB=空时硬答 val）。
    # 跳过时 summary 的 initial_prompt 字段记 null。
    initial_baseline: list[dict] | None = None
    if skip_initial_baseline:
        print("\n[Phase 0] initial-prompt baseline: skipped (--skip-initial-baseline)")
    else:
        initial_baseline = engine.run_initial_baseline(eval_set)

    print(f"\n########## Phase 1: Training & Reflection Loop ({n_iters} iters) ##########")
    history = engine.train_and_reflect(train, n_iters=n_iters, verbose=True,
                                       val_set=val)

    print(f"\n########## Phase 2: Inference eval on {eval_name} ({len(eval_set)} samples) ##########")
    val_results: list[dict] = []
    correct = 0
    for i, sample in enumerate(eval_set, 1):
        answer, meta = engine.inference(sample.problem, verbose=False,
                                        sample=sample)
        # grade with the per-sample judger (mixed datasets dispatch by tag;
        # verifiable/redaction judge the FULL raw response, consistent with
        # training — grading a short extracted answer would break constraints)
        judger = engine.task_lm.dataset_judgers.get(sample.dataset) \
            or engine.task_lm.judger
        if sample.answer_type in ("verifiable", "redaction"):
            pred_for_judge = meta.get("raw", "")
        else:
            pred_for_judge = answer
        ok = judger(pred_for_judge, sample.answer) if judger else False
        correct += ok
        tag = "OK" if ok else "X"
        ds_tag = f" [{sample.dataset}]" if "+" in dataset else ""
        print(f"  [{i}/{len(eval_set)}]{ds_tag} {tag} | branch={meta['branch']}"
              f" | pred={answer[:40]!r} | gold={sample.answer[:40]!r}")
        val_results.append({
            "index": i,
            "problem": sample.problem,
            "prediction": answer,
            "gold": sample.answer,
            "correct": bool(ok),
            "dataset": sample.dataset,
            "branch": meta["branch"],
            "dynamic_added": meta.get("dynamic_added", False),
            "matched_examples": meta.get("matched_examples", []),
            "raw": meta.get("raw", ""),
        })
    print(f"\n{eval_name} accuracy: {correct}/{len(eval_set)} = {correct / len(eval_set):.2%}")

    # Phase 2b：no-pattern 复测（同一 strategy，关 pattern 注入和动态学习）。
    # 仅 test_use_patterns=True 时有意义（否则 Phase 2 已是纯策略硬答，复测结果相同）。
    no_pattern_rerun: list[dict] | None = None
    rerun_on = (cfg.test_no_pattern_rerun
                if test_no_pattern_rerun is None
                else test_no_pattern_rerun)
    if rerun_on and not test_use_patterns:
        print("\n[Phase 2b] no-pattern rerun ignored: test_use_patterns is off "
              "(Phase 2 already answers with strategy only)")
    elif rerun_on:
        print(f"\n########## Phase 2b: No-pattern rerun on {eval_name} ##########")
        no_pattern_rerun = engine.run_eval_no_patterns(eval_set)

    # ---- save outputs ----
    run_params = {
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "eval_on": eval_name, "n_iters": n_iters,
        "seed": seed, "dynamic_learning": dynamic_learning,
        "test_use_patterns": test_use_patterns,
        "test_no_pattern_rerun": rerun_on,
        "extract_patterns_on_reject": engine.extract_patterns_on_reject,
        "evo_mode": evo_mode, "train_retrieve_ctx": train_retrieve_ctx,
        "gepa_split": gepa_split,
        "pattern_gen_mode": engine.pattern_gen_mode,
        "test_match_method": engine.test_match_method,
        "reflect_wrong_only": engine.reflect_wrong_only,
        "regularized_verify": engine.regularized_verify,
        "npo_window": engine.npo_window,
        "cross_aware_gen": engine.cross_aware_gen,
        "initial_prompt": engine.initial_prompt if engine.initial_prompt is not None else "",
        "initial_prompt_explicit": engine.initial_prompt is not None,
        # 实际生效值（CLI 覆盖后的），非 .env 快照
        "task_temperature": engine.task_lm.temperature,
        "reflection_temperature": engine.reflection_lm.temperature,
        "task_max_tokens": engine.task_lm.max_tokens,
        "reflection_max_tokens": engine.reflection_lm.max_tokens,
        "task_thinking_budget": engine.task_lm.thinking_budget,
        "reflection_thinking_budget": engine.reflection_lm.thinking_budget,
    }
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sro_output_{dataset.replace('+', '_')}_{ts}"
    _save_outputs(Path(output_dir), dataset, cfg, run_params, history,
                  val_results, engine, initial_baseline, no_pattern_rerun)


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="SRO: two-phase reflective self-evolution framework",
    )
    parser.add_argument("--demo", action="store_true", help="run built-in placeholder demo")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset to run: gsm8k / math / math500 / aime / "
                             "hotpotqa / ifbench / livebench / pupa, or a mixed "
                             "pair 'a+b' (exactly 2, e.g. gsm8k+math500; "
                             "per-sample graders/formats dispatched by tag)")
    # these default to .env values; CLI overrides when provided.
    # n-train/n-val/n-test accept int OR "a,b" string for mixed datasets:
    # single number a = take a from EACH member dataset; "a,b" = a from
    # dataset 1, b from dataset 2. Validated in run_dataset / load.
    parser.add_argument("--n-train", type=str, default=str(cfg.n_train),
                        help=f"train samples: integer, or 'a,b' per-dataset quotas for "
                             f"mixed 'A+B' (single a = a from EACH dataset) "
                             f"(default from .env: {cfg.n_train})")
    parser.add_argument("--n-val", type=str, default=str(cfg.n_val),
                        help=f"val samples: integer or 'a,b' (same semantics as --n-train) "
                             f"(default from .env: {cfg.n_val})")
    parser.add_argument("--n-test", type=str, default=str(cfg.n_test),
                        help=f"test samples: integer or 'a,b' (same semantics as --n-train); "
                             f"ignored when --gepa-split (default from .env: {cfg.n_test})")
    parser.add_argument("--n-iters", type=int, default=cfg.n_iters,
                        help=f"number of training iterations (default from .env: {cfg.n_iters})")
    parser.add_argument("--seed", type=int, default=cfg.seed,
                        help=f"random seed (default from .env: {cfg.seed})")
    parser.add_argument("--dynamic-learning", action="store_true",
                        default=cfg.dynamic_learning,
                        help="enable dynamic learning on miss (default from .env)")
    parser.add_argument("--no-dynamic", dest="dynamic_learning", action="store_false",
                        help="disable dynamic learning on miss")
    parser.add_argument("--test-use-patterns", action="store_true",
                        default=cfg.test_use_patterns,
                        help="inject KB short-term patterns as context at test time (default from .env)")
    parser.add_argument("--no-test-patterns", dest="test_use_patterns", action="store_false",
                        help="disable injecting short-term patterns as context at test time")
    parser.add_argument("--evo-mode", choices=["gepa", "classic", "npo"],
                        default=cfg.evo_mode,
                        help=f"evolution mode: classic=full-train rounds / gepa=Pareto+budget / "
                             f"npo=single-lineage sliding-window teacher revision "
                             f"(default from .env: {cfg.evo_mode})")
    parser.add_argument("--npo-window", type=int, default=None,
                        help="NPO sliding-window size W: number of recent prompt versions the "
                             "teacher sees (default from .env: NPO_WINDOW; W=1 degrades to "
                             "memoryless single-round reflection)")
    parser.add_argument("--train-retrieve-ctx", action="store_true",
                        default=cfg.train_retrieve_ctx,
                        help="retrieve KB patterns during training (default from .env)")
    parser.add_argument("--no-train-ctx", dest="train_retrieve_ctx",
                        action="store_false",
                        help="disable KB retrieval during training")
    parser.add_argument("--gepa-split", action="store_true",
                        default=cfg.gepa_split,
                        help="all datasets: replicate GEPA-protocol splits, ignoring --n-train/--n-val/--seed "
                             "(fixed seed=0; aime=half/half+### prefix, hotpotqa/gsm8k/math/math500/ifbench=150/300, "
                             "livebench=3-way equal ~133/133, pupa=positional 111/111)")
    parser.add_argument("--pattern-gen-mode", choices=["basic", "rich"],
                        default=None,
                        help="short-term pattern generation mode: basic=max 6 patterns/reflect (default from .env), "
                             "rich=multi-angle extraction + error clustering (~6x more patterns, ~6x reflection cost)")
    parser.add_argument("--test-match-method", choices=["vector", "llm", "random"],
                        default=None,
                        help="test-time pattern matching: vector=cosine only (default from .env), "
                             "llm=vector recall + LLM applicability judge (retrieve-then-rerank, +1 LLM call per test question), "
                             "random=vector recall top-N + random k (no applicability judge; ablation)")
    parser.add_argument("--reflect-wrong-only", action="store_true",
                        default=None,
                        help="STEVE-style error-driven reflection: extract lessons/diagnostics ONLY from "
                             "wrong traces, filtering out noisy gradients from already-correct examples "
                             "(default from .env: REFLECT_WRONG_ONLY)")
    parser.add_argument("--regularized-verify", action="store_true",
                        default=None,
                        help="STEVE regularized verification: gate candidate acceptance on the preservation "
                             "set (initial-correct val samples), rejecting updates whose minibatch gain "
                             "is outweighed by lambda_t * preservation regressions, lambda_t=1.5+0.1t "
                             "(default from .env: REGULARIZED_VERIFY; GEPA mode only)")
    parser.add_argument("--cross-aware-gen", action="store_true",
                        default=None,
                        help="cross-aware generation: pattern-extraction prompts see the current "
                             "long-term strategy, strategy-reflection prompts see existing KB "
                             "patterns; each side is told what the other already covers, so "
                             "patterns stay instance-level and strategy stays methodology-level "
                             "(avoid content overlap) (default from .env: CROSS_AWARE_GEN)")
    parser.add_argument("--no-cross-aware-gen", dest="cross_aware_gen",
                        action="store_false",
                        help="disable cross-aware generation (independent reflection prompts)")
    parser.add_argument("--initial-prompt", type=str, default=None,
                        help="override the default starting strategy text for task_lm "
                             "(e.g. GEPA's original INITIAL_WRITER_PROMPT). NOT passing it "
                             "(and empty .env INITIAL_PROMPT) = per-mode default (classic: "
                             "empty strategy, gepa/npo: generic fallback). Passing an "
                             "explicitly EMPTY or whitespace string (e.g. --initial-prompt "
                             "' ') = force an empty starting strategy: no gepa/npo fallback "
                             "prompt, task_lm gets only the dataset format instruction. "
                             "Works with --cross-aware-gen: pattern-extraction prompts see "
                             "this text as the current strategy from iteration 1 "
                             "(default from .env: INITIAL_PROMPT; env blank = not set)")
    parser.add_argument("--test-no-pattern-rerun", action="store_true",
                        default=None,
                        help="after the normal Phase 2 eval, re-run test once more with the SAME "
                             "final strategy but patterns/dynamic-learning OFF (isolates how much "
                             "pattern injection contributes); only meaningful with "
                             "--test-use-patterns, ignored with a warning otherwise "
                             "(default from .env: TEST_NO_PATTERN_RERUN)")
    parser.add_argument("--no-test-no-pattern-rerun", dest="test_no_pattern_rerun",
                        action="store_false",
                        help="disable the no-pattern rerun pass")
    parser.add_argument("--extract-patterns-on-reject", action="store_true",
                        default=None,
                        help="GEPA mode: still extract short-term patterns into KB from "
                             "REJECTED candidates (strict-improvement or STEVE-gate rejects); "
                             "maximizes experience reuse when training data is scarce, at the "
                             "risk of low-quality patterns (pair with high "
                             "PATTERN_DEDUP_THRESHOLD). Strategy is NOT updated on reject "
                             "(default from .env: EXTRACT_PATTERNS_ON_REJECT)")
    parser.add_argument("--no-extract-patterns-on-reject",
                        dest="extract_patterns_on_reject", action="store_false",
                        help="only extract patterns from accepted candidates (vanilla GEPA)")
    parser.add_argument("--task-temperature", type=float, default=None,
                        help=f"TaskLM sampling temperature (default from .env: "
                             f"{cfg.task_temperature}; 0.0=deterministic)")
    parser.add_argument("--reflection-temperature", type=float, default=None,
                        help=f"ReflectionLM sampling temperature (default from .env: "
                             f"{cfg.reflection_temperature})")
    parser.add_argument("--task-max-tokens", type=int, default=None,
                        help=f"TaskLM max output tokens per call (default from .env: "
                             f"{cfg.task_max_tokens})")
    parser.add_argument("--reflection-max-tokens", type=int, default=None,
                        help=f"ReflectionLM max output tokens per call (default from .env: "
                             f"{cfg.reflection_max_tokens})")
    parser.add_argument("--task-thinking-budget", type=int, default=None,
                        help=f"TaskLM thinking budget in tokens (vLLM endpoints only; "
                             f"0=unlimited, total capped by --task-max-tokens; >0 forces "
                             f"answer after budget via Qwen3 official early-close injection) "
                             f"(default from .env: {cfg.task_thinking_budget})")
    parser.add_argument("--reflection-thinking-budget", type=int, default=None,
                        help=f"ReflectionLM thinking budget (same semantics as "
                             f"--task-thinking-budget) (default from .env: "
                             f"{cfg.reflection_thinking_budget})")
    parser.add_argument("--skip-initial-baseline", action="store_true",
                        help="skip the Phase 0 initial-prompt baseline on val "
                             "(saves |val| extra LLM calls; summary fields become null)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory (default: sro_output_<dataset>_<timestamp>)")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset:
        run_dataset(args.dataset, args.n_train, args.n_val, args.n_test,
                    args.n_iters,
                    args.seed, args.dynamic_learning, args.output_dir,
                    args.evo_mode, args.train_retrieve_ctx, args.test_use_patterns,
                    args.gepa_split, args.pattern_gen_mode,
                    args.test_match_method, args.reflect_wrong_only,
                    args.regularized_verify, args.npo_window,
                    args.cross_aware_gen, args.initial_prompt,
                    args.test_no_pattern_rerun,
                    args.extract_patterns_on_reject,
                    args.task_temperature, args.reflection_temperature,
                    args.task_max_tokens, args.reflection_max_tokens,
                    args.task_thinking_budget, args.reflection_thinking_budget,
                    args.skip_initial_baseline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
