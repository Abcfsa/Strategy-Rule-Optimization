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
    }
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
    evo_mode: str, train_retrieve_ctx: bool,
) -> None:
    """Load a real dataset and run the two-phase loop, then save outputs.

    Phase 1: reflect-and-iterate on train; Phase 2: inference + eval on val.
    """
    from sro.datasets import load

    cfg = get_config()
    if not cfg.has_api_key:
        print("[WARN] OPENAI_API_KEY not set; using placeholder LM (data-flow demo only).")

    print(f"########## Loading dataset: {dataset} ##########")
    train, val = load(dataset, n_train=n_train, n_val=n_val, seed=seed)
    print(f"  train: {len(train)} samples | val: {len(val)} samples")

    engine = SROEngine(
        match_threshold=cfg.match_threshold, top_k=cfg.top_k,
        dynamic_learning=dynamic_learning,
        evo_mode=evo_mode, train_retrieve_ctx=train_retrieve_ctx,
        max_metric_calls=cfg.max_metric_calls, minibatch_size=cfg.minibatch_size,
        max_prompt_length=cfg.max_prompt_length, seed=seed,
    )
    engine.set_dataset(dataset)   # inject the matching grader

    print(f"\n########## Phase 1: Training & Reflection Loop ({n_iters} iters) ##########")
    history = engine.train_and_reflect(train, n_iters=n_iters, verbose=True)

    print(f"\n########## Phase 2: Inference eval on val ({len(val)} samples) ##########")
    val_results: list[dict] = []
    correct = 0
    for i, sample in enumerate(val, 1):
        answer, meta = engine.inference(sample.problem, verbose=False)
        # grade with the injected judger (consistent with training)
        ok = engine.task_lm.judger(answer, sample.answer) if engine.task_lm.judger else False
        correct += ok
        tag = "OK" if ok else "X"
        print(f"  [{i}/{len(val)}] {tag} | branch={meta['branch']}"
              f" | pred={answer[:40]!r} | gold={sample.answer[:40]!r}")
        val_results.append({
            "index": i,
            "problem": sample.problem,
            "prediction": answer,
            "gold": sample.answer,
            "correct": bool(ok),
            "branch": meta["branch"],
            "dynamic_added": meta.get("dynamic_added", False),
            "matched_examples": meta.get("matched_examples", []),
            "raw": meta.get("raw", ""),
        })
    print(f"\nval accuracy: {correct}/{len(val)} = {correct / len(val):.2%}")

    # ---- save outputs ----
    run_params = {
        "n_train": n_train, "n_val": n_val, "n_iters": n_iters,
        "seed": seed, "dynamic_learning": dynamic_learning,
        "evo_mode": evo_mode, "train_retrieve_ctx": train_retrieve_ctx,
    }
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"sro_output_{dataset}_{ts}"
    _save_outputs(Path(output_dir), dataset, cfg, run_params, history, val_results, engine)


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="SRO: two-phase reflective self-evolution framework",
    )
    parser.add_argument("--demo", action="store_true", help="run built-in placeholder demo")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "aime", "hotpotqa"],
                        help="load a real dataset and run the two-phase loop")
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
    parser.add_argument("--evo-mode", choices=["gepa", "classic"],
                        default=cfg.evo_mode,
                        help=f"evolution mode (default from .env: {cfg.evo_mode})")
    parser.add_argument("--train-retrieve-ctx", action="store_true",
                        default=cfg.train_retrieve_ctx,
                        help="retrieve KB patterns during training (default from .env)")
    parser.add_argument("--no-train-ctx", dest="train_retrieve_ctx",
                        action="store_false",
                        help="disable KB retrieval during training")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="output directory (default: sro_output_<dataset>_<timestamp>)")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset:
        run_dataset(args.dataset, args.n_train, args.n_val, args.n_iters,
                    args.seed, args.dynamic_learning, args.output_dir,
                    args.evo_mode, args.train_retrieve_ctx)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
