"""Command Line Interface (CLI) for running AutoResearcher experiments and benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoresearcher.benchmark.runner import (
    generate_markdown_report,
    run_full_benchmark,
    run_single_strategy,
)
from autoresearcher.config import DEFAULT_BENCHMARK_DATASETS, SETTINGS
from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoresearcher",
        description="AutoResearcher: Autonomous Scientific ML Experimentation Agent & Benchmark Suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run-agent command
    p_agent = subparsers.add_parser("run-agent", help="Run the AutoResearcher Agent on a single dataset")
    p_agent.add_argument("--dataset", type=str, default="credit-g", help="Dataset name (credit-g, phoneme, vehicle)")
    p_agent.add_argument("--budget", type=int, default=15, help="Number of experiment trials (default: 15)")
    p_agent.add_argument("--seed", type=int, default=42, help="Random seed")
    p_agent.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds")
    p_agent.add_argument("--offline", action="store_true", help="Use deterministic offline dataset generator")

    # 2. run-benchmark command
    p_bench = subparsers.add_parser(
        "run-benchmark",
        help="Run full comparative benchmark (Agent vs Optuna TPE vs Random vs Grid)",
    )
    p_bench.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_BENCHMARK_DATASETS,
        help="Datasets to include in benchmark",
    )
    p_bench.add_argument("--budget", type=int, default=15, help="Trial budget per run (default: 15)")
    p_bench.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 101, 202],
        help="Seeds for stochastic baselines (default: 42 101 202)",
    )
    p_bench.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds (default: 5)")
    p_bench.add_argument("--offline", action="store_true", help="Use deterministic offline fallback datasets")

    # 3. generate-report command
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
            marker = "★ NEW BEST" if rec.was_new_best else " "
            print(
                f"[Trial {rec.trial_index:02d}] Val F1={rec.diagnostics.val_macro_f1:.4f} "
                f"(Best={rec.best_val_macro_f1_so_far:.4f}) | Gap={rec.diagnostics.overfit_gap:+.4f} "
                f"| Model={rec.config.model_family} {marker}"
            )
            if rec.agent_decision:
                print(f"   Diagnosis : {rec.agent_decision.bottleneck_diagnosis}")
                print(f"   Hypothesis: {rec.agent_decision.scientific_hypothesis}")

        res = run_single_strategy(
            strategy_type="agent",
            dataset=ds,
            budget=args.budget,
            seed=args.seed,
            n_splits=args.folds,
            tracker=tracker,
            progress_callback=_print_step,
        )
        print(
            f"\nFinal Result on '{args.dataset}': Best CV Macro-F1 = {res.best_val_macro_f1:.4f} "
            f"| Locked Holdout Test Macro-F1 = {res.holdout_test_macro_f1:.4f}"
        )

    elif args.command == "run-benchmark":
        run_full_benchmark(
            datasets=args.datasets,
            budget=args.budget,
            stochastic_seeds=args.seeds,
            n_splits=args.folds,
            prefer_offline_fallback=args.offline,
        )

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
