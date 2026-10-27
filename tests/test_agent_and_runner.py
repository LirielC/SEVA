"""End-to-end tests for AutoResearcherAgent, anti-duplication, and comparative benchmark runner."""

from __future__ import annotations

from pathlib import Path

from autoresearcher.benchmark.runner import generate_markdown_report, run_single_strategy
from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def test_agent_never_repeats_configs_and_improves_over_baseline(tmp_path: Path) -> None:
    ds = load_benchmark_dataset("credit-g", split_seed=42, prefer_offline_fallback=True)
    tracker = ExperimentTracker(artifacts_dir=tmp_path, enable_mlflow=False)

    res_agent = run_single_strategy(
        strategy_type="agent",
        dataset=ds,
        budget=6,
        seed=42,
        n_splits=3,
        tracker=tracker,
    )

    assert len(res_agent.trials) == 6
    hashes = [t.config_hash for t in res_agent.trials]
    # Anti-duplication check: all 6 evaluated configurations must be strictly distinct
    assert len(set(hashes)) == 6

    # Every agent trial must contain a structured scientific decision
    for t in res_agent.trials:
        assert t.agent_decision is not None
        assert len(t.agent_decision.bottleneck_diagnosis) > 10
        assert len(t.agent_decision.scientific_hypothesis) > 10

    # Agent should improve over the unregularized Trial 0 baseline on credit-g
    assert res_agent.best_val_macro_f1 >= res_agent.baseline_val_macro_f1

    # Also test Optuna and Random on the same dataset and generate report + plots
    res_opt = run_single_strategy(
        strategy_type="optuna",
        dataset=ds,
        budget=6,
        seed=42,
        n_splits=3,
        tracker=tracker,
    )
    res_rnd = run_single_strategy(
        strategy_type="random",
        dataset=ds,
        budget=6,
        seed=42,
        n_splits=3,
        tracker=tracker,
    )

    all_res = [res_agent, res_opt, res_rnd]
    plots = generate_benchmark_plots(all_res, output_dir=tmp_path / "plots")
    assert len(plots) == 2
    for p in plots:
        assert p.exists()

    report_md = generate_markdown_report(all_res, output_path=tmp_path / "BENCHMARK_REPORT.md")
    assert "AutoResearcher (Agent)" in report_md
    assert "Optuna (TPE)" in report_md
