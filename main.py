"""SRO entry point.

- `python main.py --demo`          : built-in placeholder demo (no API key / dataset)
- `python main.py --dataset NAME`  : load a real dataset, run two-phase loop, save outputs

Datasets: gsm8k / math / aime / hotpotqa (need local data + API key).

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
) -> None:
    """Save run artifacts to out_dir (mirrors gepa_aime_v3 multi-file output).

    run_params: the actual values used this run (CLI overrides applied),
    so config.json/summary.json reflect what executed, not the .env defaults.
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

    # summary.json — high-level training + val results (uses actual run params)
    val_correct = sum(1 for r in val_results if r["correct"])
    # per-dataset accuracy buckets (mixed-dataset runs; {} for single dataset)
    per_ds: dict[str, dict] = {}
    for r in val_results:
        ds = r.get("dataset", "")
        if ds:
            b = per_ds.setdefault(ds, {"correct": 0, "total": 0})
            b["total"] += 1
            b["correct"] += 1 if r["correct"] else 0
    for b in per_ds.values():
        b["accuracy"] = b["correct"] / b["total"] if b["total"] else 0.0
    summary = {
        "dataset": dataset,
        "evo_mode": run_params["evo_mode"],
        "n_train": run_params["n_train"],
        "n_val": run_params["n_val"],
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
        "per_dataset_accuracy": per_ds,   # {} for single-dataset runs
    }
    # 初始 prompt 基线（Phase 0）：训练前的 val 成绩 + 最终 vs 初始的提升。
    # 混合模式同样做 per-dataset 分桶（与 per_dataset_accuracy 口径一致）。
    if initial_baseline:
        base_correct = sum(1 for r in initial_baseline if r["correct"])
        base_per_ds: dict[str, dict] = {}
        for r in initial_baseline:
            ds = r.get("dataset", "")
            if ds:
                b = base_per_ds.setdefault(ds, {"correct": 0, "total": 0})
                b["total"] += 1
                b["correct"] += 1 if r["correct"] else 0
        for b in base_per_ds.values():
            b["accuracy"] = b["correct"] / b["total"] if b["total"] else 0.0
        summary["initial_prompt"] = {
            "correct": base_correct,
            "total": len(initial_baseline),
            "accuracy": (base_correct / len(initial_baseline)
                         if initial_baseline else 0.0),
            "per_dataset_accuracy": base_per_ds,   # {} for single-dataset runs
        }
        if val_results:
            summary["val_improvement_vs_initial"] = (
                val_correct / len(val_results) - base_correct / len(initial_baseline))
            # per-dataset 提升（同名桶两侧都存在时才报，避免除零/空桶）
            if per_ds and base_per_ds:
                summary["improvement_per_dataset"] = {
                    ds: per_ds[ds]["accuracy"] - base_per_ds[ds]["accuracy"]
                    for ds in per_ds if ds in base_per_ds and per_ds[ds]["total"]
                }
    else:
        summary["initial_prompt"] = None
        summary["val_improvement_vs_initial"] = None
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


def run_dataset(
    dataset: str, n_train: int, n_val: int, n_iters: int,
    seed: int, dynamic_learning: bool, output_dir: str | None,
    evo_mode: str, train_retrieve_ctx: bool, test_use_patterns: bool,
    aime_gepa_split: bool = False, pattern_gen_mode: str | None = None,
    test_match_method: str | None = None, reflect_wrong_only: bool | None = None,
    regularized_verify: bool | None = None, npo_window: int | None = None,
    skip_initial_baseline: bool = False,
) -> None:
    """Load a real dataset and run the two-phase loop, then save outputs.

    Phase 0 (optional): initial-prompt baseline on val; Phase 1: reflect-and-iterate
    on train; Phase 2: inference + eval on val.
    """
    from sro.datasets import load

    cfg = get_config()
    if not cfg.has_api_key:
        print("[WARN] OPENAI_API_KEY not set; using placeholder LM (data-flow demo only).")

    print(f"########## Loading dataset: {dataset} ##########")
    train, val = load(dataset, n_train=n_train, n_val=n_val, seed=seed,
                      gepa_split=aime_gepa_split)
    print(f"  train: {len(train)} samples | val: {len(val)} samples")

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
    )
    engine.set_dataset(dataset, gepa_split=aime_gepa_split)   # inject the matching grader

    # Phase 0：初始 prompt 基线（策略=初始版本、KB=空时硬答 val）。
    # 混合模式下逐样本 judger/format 分发生效。跳过时 summary 记 null。
    initial_baseline: list[dict] | None = None
    if skip_initial_baseline:
        print("\n[Phase 0] initial-prompt baseline: skipped (--skip-initial-baseline)")
    else:
        initial_baseline = engine.run_initial_baseline(val)

    print(f"\n########## Phase 1: Training & Reflection Loop ({n_iters} iters) ##########")
    history = engine.train_and_reflect(train, n_iters=n_iters, verbose=True)

    print(f"\n########## Phase 2: Inference eval on val ({len(val)} samples) ##########")
    val_results: list[dict] = []
    correct = 0
    for i, sample in enumerate(val, 1):
        # sample passed through for mixed-mode per-sample judger/format dispatch
        answer, meta = engine.inference(sample.problem, verbose=False,
                                        sample=sample)
        # unified grading entry: mixed -> per-sample judger; single -> injected
        ok = engine.grade(sample, answer)
        correct += ok
        tag = "OK" if ok else "X"
        ds_tag = f" | ds={sample.dataset}" if sample.dataset else ""
        print(f"  [{i}/{len(val)}] {tag} | branch={meta['branch']}{ds_tag}"
              f" | pred={answer[:40]!r} | gold={sample.answer[:40]!r}")
        val_results.append({
            "index": i,
            "problem": sample.problem,
            "prediction": answer,
            "gold": sample.answer,
            "correct": bool(ok),
            "branch": meta["branch"],
            "dataset": sample.dataset,   # "" for single-dataset runs
            "dynamic_added": meta.get("dynamic_added", False),
            "matched_examples": meta.get("matched_examples", []),
            "raw": meta.get("raw", ""),
        })
    print(f"\nval accuracy: {correct}/{len(val)} = {correct / len(val):.2%}")

    # per-dataset bucketed accuracy (the core readout for mixed-dataset runs;
    # single-dataset runs yield one bucket or none)
    buckets: dict[str, list[bool]] = {}
    for r in val_results:
        if r["dataset"]:
            buckets.setdefault(r["dataset"], []).append(r["correct"])
    if buckets:
        print("per-dataset accuracy:")
        for ds, flags in sorted(buckets.items()):
            print(f"  {ds}: {sum(flags)}/{len(flags)}"
                  f" = {sum(flags) / len(flags):.2%}")

    # ---- save outputs ----
    run_params = {
        "n_train": n_train, "n_val": n_val, "n_iters": n_iters,
        "seed": seed, "dynamic_learning": dynamic_learning,
        "test_use_patterns": test_use_patterns,
        "evo_mode": evo_mode, "train_retrieve_ctx": train_retrieve_ctx,
        "aime_gepa_split": aime_gepa_split,
        "pattern_gen_mode": engine.pattern_gen_mode,
        "test_match_method": engine.test_match_method,
        "reflect_wrong_only": engine.reflect_wrong_only,
        "regularized_verify": engine.regularized_verify,
        "npo_window": engine.npo_window,
        "mixed_datasets": engine.mixed_names,   # [] for single-dataset runs
    }
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sro_output_{dataset}_{ts}"
    _save_outputs(Path(output_dir), dataset, cfg, run_params, history,
                  val_results, engine, initial_baseline)


_VALID_DATASETS = ["gsm8k", "math", "aime", "hotpotqa"]


def _validate_dataset(name: str) -> str:
    """Accept a single dataset name or a mixed 'a+b' combination.

    Mixed mode: 2+ known datasets joined by '+'; aime may appear as one of
    the members (with --aime-gepa-split it takes the GEPA protocol side).
    """
    if "+" in name:
        parts = [p.strip() for p in name.split("+") if p.strip()]
        if len(parts) < 2:
            raise argparse.ArgumentTypeError(
                f"mixed dataset needs >=2 names: {name!r}")
        bad = [p for p in parts if p not in _VALID_DATASETS]
        if bad:
            raise argparse.ArgumentTypeError(
                f"unknown dataset(s) {bad}; choose from {_VALID_DATASETS}")
        return "+".join(parts)
    if name not in _VALID_DATASETS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {name!r}; choose from {_VALID_DATASETS}"
            f" or a mixed 'a+b' combination")
    return name


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="SRO: two-phase reflective self-evolution framework",
    )
    parser.add_argument("--demo", action="store_true", help="run built-in placeholder demo")
    parser.add_argument("--dataset", type=_validate_dataset,
                        help="load a real dataset and run the two-phase loop; "
                             "single: gsm8k/math/aime/hotpotqa, or mixed 'a+b' "
                             "(e.g. gsm8k+hotpotqa; with --aime-gepa-split, "
                             "aime members use the GEPA protocol)")
    # these default to .env values; CLI overrides when provided
    parser.add_argument("--n-train", type=int, default=cfg.n_train,
                        help=f"number of train samples (default from .env: {cfg.n_train})")
    parser.add_argument("--n-val", type=int, default=cfg.n_val,
                        help=f"number of val samples (default from .env: {cfg.n_val})")
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
    parser.add_argument("--aime-gepa-split", action="store_true",
                        default=cfg.aime_gepa_split,
                        help="AIME only: replicate GEPA init_dataset() split (seed=0, half/half, ### answer prefix). "
                             "In mixed mode, applies only to the aime member; other members keep their seeded splits")
    parser.add_argument("--pattern-gen-mode", choices=["basic", "rich"],
                        default=None,
                        help="short-term pattern generation mode: basic=max 6 patterns/reflect (default from .env), "
                             "rich=multi-angle extraction + error clustering (~6x more patterns, ~6x reflection cost)")
    parser.add_argument("--test-match-method", choices=["vector", "llm"],
                        default=None,
                        help="test-time pattern matching: vector=cosine only (default from .env), "
                             "llm=vector recall + LLM applicability judge (retrieve-then-rerank, +1 LLM call per test question)")
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
    parser.add_argument("--skip-initial-baseline", action="store_true",
                        help="skip the Phase 0 initial-prompt baseline on val "
                             "(saves |val| extra LLM calls; summary fields become null)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory (default: sro_output_<dataset>_<timestamp>)")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset:
        run_dataset(args.dataset, args.n_train, args.n_val, args.n_iters,
                    args.seed, args.dynamic_learning, args.output_dir,
                    args.evo_mode, args.train_retrieve_ctx, args.test_use_patterns,
                    args.aime_gepa_split, args.pattern_gen_mode,
                    args.test_match_method, args.reflect_wrong_only,
                    args.regularized_verify, args.npo_window,
                    args.skip_initial_baseline)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
