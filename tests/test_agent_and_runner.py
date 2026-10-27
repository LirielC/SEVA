"""Comprehensive tests for SEVA 4-way strategy ablation, scientific traceability, and replay reproducibility."""

from __future__ import annotations

from pathlib import Path

from autoresearcher.benchmark.runner import (
    generate_markdown_report,
    replay_pipeline_from_config,
    run_single_strategy,
)
from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def test_scientific_traceability_and_expected_vs_observed_outcome(tmp_path: Path) -> None:
    """Verifies that every trial records diagnosis, hypothesis, expected outcome, and observed verification."""
    ds = load_benchmark_dataset("credit-g", split_seed=42, prefer_offline_fallback=True)
    tracker = ExperimentTracker(artifacts_dir=tmp_path, enable_mlflow=False)

    res_agent = run_single_strategy(
        strategy_type="heuristic_agent",
        dataset=ds,
        budget=5,
        seed=42,
        n_splits=3,
        tracker=tracker,
    )

    assert len(res_agent.trials) == 5
    assert res_agent.auc_convergence > 0.5
    assert res_agent.trials_to_95pct_peak >= 0

    hashes = [t.config_hash for t in res_agent.trials]
    assert len(set(hashes)) == 5  # Zero duplicates

    for t in res_agent.trials:
        # Check scientific method schema completeness
        assert t.agent_decision is not None
        assert len(t.agent_decision.bottleneck_diagnosis) > 10
        assert len(t.agent_decision.scientific_hypothesis) > 10
        assert len(t.agent_decision.expected_outcome) > 5

        # Check post-evaluation observed outcome verification
        assert t.observed_outcome is not None
        assert len(t.observed_outcome) > 10
        assert isinstance(t.hypothesis_confirmed, bool)


def test_four_strategy_ablation_and_report_generation(tmp_path: Path) -> None:
    """Ablation study comparing all 4 strategies under identical budget and metrics."""
    ds = load_benchmark_dataset("credit-g", split_seed=42, prefer_offline_fallback=True)
    tracker = ExperimentTracker(artifacts_dir=tmp_path, enable_mlflow=False)

    strategies = ["heuristic_agent", "llm_agent", "optuna", "random"]
    results = []

    for strat in strategies:
        res = run_single_strategy(
            strategy_type=strat,
            dataset=ds,
            budget=4,
            seed=42,
            n_splits=3,
            target_threshold=0.88,
            tracker=tracker,
        )
        assert len(res.trials) == 4
        assert res.auc_convergence > 0.0
        results.append(res)

    plots = generate_benchmark_plots(results, output_dir=tmp_path / "plots")
    assert len(plots) == 2
    for p in plots:
        assert p.exists()

    rep = generate_markdown_report(results, output_path=tmp_path / "BENCHMARK_REPORT.md")
    assert "SEVA (Heuristic Agent)" in rep
    assert "SEVA (LLM Agent)" in rep
    assert "Optuna (TPE)" in rep
    assert "Random Search" in rep
    assert "AUC Convergence" in rep


def test_deterministic_reproducibility_replay(tmp_path: Path) -> None:
    """Verifies that any pipeline can be reproducibly re-evaluated identically from seed and config."""
    ds = load_benchmark_dataset("credit-g", split_seed=42, prefer_offline_fallback=True)
    tracker = ExperimentTracker(artifacts_dir=tmp_path, enable_mlflow=False)

    res = run_single_strategy(
        strategy_type="heuristic_agent",
        dataset=ds,
        budget=4,
        seed=42,
        n_splits=3,
        tracker=tracker,
    )
    best_config = res.best_config

    # Replay using the replay function
    replay_out = replay_pipeline_from_config(
        config_source=best_config.model_dump(),
        dataset_name="credit-g",
        seed=42,
        n_splits=3,
        prefer_offline_fallback=True,
    )

    # Replay must yield exact same hash, CV score, and holdout test score
    assert replay_out["config_hash"] == best_config.config_hash()
    assert abs(replay_out["cv_val_macro_f1"] - res.best_val_macro_f1) < 1e-4
    assert abs(replay_out["holdout_test_macro_f1"] - res.holdout_test_macro_f1) < 1e-4
