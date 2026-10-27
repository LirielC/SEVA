"""Command Line Interface (CLI) for SEVA: Autonomous Scientific ML Agent & Benchmark Suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoresearcher.benchmark.runner import (
    generate_markdown_report,
    replay_pipeline_from_config,
    run_full_benchmark,
    run_single_strategy,
)
from autoresearcher.config import DEFAULT_BENCHMARK_DATASETS, SETTINGS
from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seva",
        description="SEVA: Secure Experimentation and Validation Agent & 4-Way Benchmark Suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run-agent command
    p_agent = subparsers.add_parser("run-agent", help="Run SEVA Agent on a single dataset")
    p_agent.add_argument("--dataset", type=str, default="credit-g", help="Dataset name (credit-g, phoneme, vehicle)")
    p_agent.add_argument("--budget", type=int, default=15, help="Number of experiment trials (default: 15)")
    p_agent.add_argument(
        "--mode",
        type=str,
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Decision engine mode: 'heuristic' (diagnostic engine) or 'llm' (Qwen2.5/OpenAI endpoint)",
    )
    p_agent.add_argument("--seed", type=int, default=42, help="Random seed")
    p_agent.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds")
    p_agent.add_argument("--target-threshold", type=float, default=None, help="Optional Macro-F1 threshold target")
    p_agent.add_argument("--offline", action="store_true", help="Use deterministic offline dataset generator")

    # 2. run-benchmark command
    p_bench = subparsers.add_parser(
        "run-benchmark",
        help="Run 4-way ablation benchmark (Heuristic Agent, LLM Agent, Optuna TPE, Random Search)",
    )
    p_bench.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_BENCHMARK_DATASETS,
        help="Datasets to include in benchmark",
    )
    p_bench.add_argument("--budget", type=int, default=12, help="Trial budget per run (default: 12)")
    p_bench.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 101, 202],
        help="Seeds for repeated runs across methods (default: 42 101 202)",
    )
    p_bench.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds (default: 5)")
    p_bench.add_argument("--include-llm", action="store_true", help="Include SEVA LLM Agent in the comparison")
    p_bench.add_argument("--offline", action="store_true", help="Use deterministic offline fallback datasets")

    # 3. replay command (Reproducibility & Resource Verification)
    p_rep_cmd = subparsers.add_parser(
        "replay",
        help="Faithfully reproduce and evaluate a pipeline from a saved JSON run or configuration file",
    )
    p_rep_cmd.add_argument("--config", type=str, required=True, help="Path to JSON run file or PipelineConfig JSON string")
    p_rep_cmd.add_argument("--dataset", type=str, default="credit-g", help="Dataset name to evaluate on")
    p_rep_cmd.add_argument("--seed", type=int, default=42, help="Seed to lock evaluation split and CV folds")
    p_rep_cmd.add_argument("--folds", type=int, default=5, help="CV folds")
    p_rep_cmd.add_argument("--offline", action="store_true", help="Use offline fallback")

    # 4. generate-report command
    p_rep = subparsers.add_parser(
        "generate-report",
        help="Regenerate Markdown report and PNG plots from saved runs in artifacts/runs",
    )
    p_rep.add_argument("--artifacts-dir", type=str, default=str(SETTINGS.artifacts_dir))

    args = parser.parse_args()

    if args.command == "run-agent":
        ds = load_benchmark_dataset(
            args.dataset,
            split_seed=args.seed,
            prefer_offline_fallback=args.offline,
        )
        tracker = ExperimentTracker()

        def _print_step(rec) -> None:
            marker = "[NEW BEST]" if rec.was_new_best else " "
            print(
                f"[Trial {rec.trial_index:02d}] Val F1={rec.diagnostics.val_macro_f1:.4f} "
                f"(Best={rec.best_val_macro_f1_so_far:.4f}) | Gap={rec.diagnostics.overfit_gap:+.4f} "
                f"| Model={rec.config.model_family} {marker}"
            )
            if rec.agent_decision:
                print(f"   Diagnosis: {rec.agent_decision.bottleneck_diagnosis}")
                print(f"   Hypothesis: {rec.agent_decision.scientific_hypothesis}")
                print(f"   Expected : {rec.agent_decision.expected_outcome}")
            if rec.observed_outcome:
                print(f"   Observed : {rec.observed_outcome}")

        strategy_code = "llm_agent" if args.mode == "llm" else "heuristic_agent"
        res = run_single_strategy(
            strategy_type=strategy_code,
            dataset=ds,
            budget=args.budget,
            seed=args.seed,
            n_splits=args.folds,
            target_threshold=args.target_threshold,
            tracker=tracker,
            progress_callback=_print_step,
        )
        print(
            f"\nFinal Result on '{args.dataset}': Best CV Macro-F1 = {res.best_val_macro_f1:.4f} "
            f"| Locked Holdout Test Macro-F1 = {res.holdout_test_macro_f1:.4f} "
            f"| AUC Convergence = {res.auc_convergence:.4f} "
            f"| Peak reached at Trial #{res.trials_to_95pct_peak}"
        )
        if res.total_llm_cost_usd > 0:
            print(f"LLM Telemetry: {res.total_llm_tokens} tokens, ${res.total_llm_cost_usd:.4f} USD cost across {res.total_llm_calls} calls.")

    elif args.command == "run-benchmark":
        run_full_benchmark(
            datasets=args.datasets,
            budget=args.budget,
            stochastic_seeds=args.seeds,
            n_splits=args.folds,
            include_llm_agent=args.include_llm,
            prefer_offline_fallback=args.offline,
        )

    elif args.command == "replay":
        out = replay_pipeline_from_config(
            config_source=args.config,
            dataset_name=args.dataset,
            seed=args.seed,
            n_splits=args.folds,
            prefer_offline_fallback=args.offline,
        )
        print("\n================ REPLAY VERIFICATION ================")
        print(json.dumps(out, indent=2))
        print("=====================================================")

    elif args.command == "generate-report":
        art_dir = Path(args.artifacts_dir)
        tracker = ExperimentTracker(artifacts_dir=art_dir)
        results = tracker.load_all_results()
        if not results:
            print(f"No saved runs found in {art_dir / 'runs'}.")
            return
        generate_benchmark_plots(results, output_dir=art_dir / "plots")
        rep_path = art_dir / "BENCHMARK_REPORT.md"
        generate_markdown_report(results, output_path=rep_path)
        print(f"Report and plots generated in {art_dir}")


if __name__ == "__main__":
    main()
