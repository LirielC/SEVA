"""Benchmark runner comparing AutoResearcher Agent against Optuna (TPE), Random Search, and Grid Search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pandas as pd

from autoresearcher.agent.diagnostic_agent import AutoResearcherAgent
from autoresearcher.baselines.grid_search import GridSearchStrategy
from autoresearcher.baselines.optuna_search import OptunaTPEStrategy
from autoresearcher.baselines.random_search import RandomSearchStrategy
from autoresearcher.config import DEFAULT_BENCHMARK_DATASETS, SETTINGS
from autoresearcher.data.openml_loader import DatasetSplit, load_benchmark_dataset
from autoresearcher.pipeline.evaluator import evaluate_on_holdout, evaluate_pipeline_cv
from autoresearcher.schemas import AgentDecision, MethodBenchmarkResult, PipelineConfig, TrialRecord
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def _compute_trials_to_95pct_gain(trials: List[TrialRecord]) -> int:
    """Compute the earliest trial index that achieved >= 95% of the run's total validation gain over Trial 0."""
    if not trials:
        return 0
    base_score = trials[0].diagnostics.val_macro_f1
    peak_score = max(t.diagnostics.val_macro_f1 for t in trials)
    total_gain = peak_score - base_score
    if total_gain <= 1e-5:
        return 0
    target_threshold = base_score + 0.95 * total_gain
    for t in trials:
        if t.best_val_macro_f1_so_far >= target_threshold - 1e-6:
            return t.trial_index
    return trials[-1].trial_index


def run_single_strategy(
    strategy_type: str,
    dataset: DatasetSplit,
    budget: int = 15,
    seed: int = 42,
    n_splits: int = 5,
    tracker: Optional[ExperimentTracker] = None,
    progress_callback: Optional[Callable[[TrialRecord], None]] = None,
) -> MethodBenchmarkResult:
    """Execute one optimization strategy for `budget` trials on a locked dataset split."""
    start_total = time.perf_counter()
    n_classes = dataset.profile.n_classes

    if strategy_type == "agent":
        agent = AutoResearcherAgent(profile=dataset.profile, budget=budget, seed=seed)
        strategy_name = AutoResearcherAgent.name
    elif strategy_type == "optuna":
        optuna_strat = OptunaTPEStrategy(seed=seed, n_classes=n_classes)
        strategy_name = OptunaTPEStrategy.name
    elif strategy_type == "random":
        rand_strat = RandomSearchStrategy(seed=seed, n_classes=n_classes)
        strategy_name = RandomSearchStrategy.name
    elif strategy_type == "grid":
        grid_strat = GridSearchStrategy(n_classes=n_classes)
        strategy_name = GridSearchStrategy.name
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")

    history: List[TrialRecord] = []
    best_val_so_far: float = -1.0
    best_config: PipelineConfig = PipelineConfig.default_baseline()

    for trial_idx in range(budget):
        decision: Optional[AgentDecision] = None
        if strategy_type == "agent":
            cfg, decision = agent.suggest_next(history)
        elif strategy_type == "optuna":
            cfg = optuna_strat.suggest_next(history)
        elif strategy_type == "random":
            cfg = rand_strat.suggest_next(history)
        else:
            cfg = grid_strat.suggest_next(history)

        diag, elapsed = evaluate_pipeline_cv(
            config=cfg,
            dataset=dataset,
            n_splits=n_splits,
            seed=seed,
        )

        was_new_best = diag.val_macro_f1 > best_val_so_far
        if was_new_best:
            best_val_so_far = diag.val_macro_f1
            best_config = cfg

        record = TrialRecord(
            trial_index=trial_idx,
            strategy_name=strategy_name,
            dataset_name=dataset.profile.name,
            seed=seed,
            config=cfg,
            config_hash=cfg.config_hash(),
            diagnostics=diag,
            best_val_macro_f1_so_far=round(best_val_so_far, 5),
            elapsed_seconds=elapsed,
            agent_decision=decision,
            was_new_best=was_new_best,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    # Final evaluation ONCE on the locked 20% Holdout Test set
    holdout_f1 = evaluate_on_holdout(best_config, dataset=dataset, seed=seed)
    baseline_f1 = history[0].diagnostics.val_macro_f1
    improvement = round(best_val_so_far - baseline_f1, 5)
    trials_to_95 = _compute_trials_to_95pct_gain(history)
    total_elapsed = round(time.perf_counter() - start_total, 4)

    result = MethodBenchmarkResult(
        strategy_name=strategy_name,
        dataset_name=dataset.profile.name,
        seed=seed,
        budget=budget,
        best_val_macro_f1=round(best_val_so_far, 5),
        holdout_test_macro_f1=holdout_f1,
        baseline_val_macro_f1=baseline_f1,
        improvement_over_baseline=improvement,
        trials_to_95pct_peak=trials_to_95,
        total_elapsed_seconds=total_elapsed,
        best_config=best_config,
        trials=history,
    )

    if tracker is not None:
        tracker.log_method_run(result)

    return result


def generate_markdown_report(
    results: List[MethodBenchmarkResult],
    output_path: Path,
) -> str:
    """Create a rigorous scientific Markdown report summarizing all benchmark runs and hypotheses."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        # Compute Area Under Convergence Curve (mean best-so-far across budget)
        auc_conv = float(np.mean([t.best_val_macro_f1_so_far for t in r.trials]))
        rows.append({
            "Dataset": r.dataset_name,
            "Strategy": r.strategy_name,
            "Seed": r.seed,
            "Baseline Val F1": r.baseline_val_macro_f1,
            "Best Val F1": r.best_val_macro_f1,
            "Gain (+F1)": r.improvement_over_baseline,
            "Holdout Test F1": r.holdout_test_macro_f1,
            "Trials to 95% Gain": r.trials_to_95pct_peak,
            "Mean Best-So-Far (AUC)": round(auc_conv, 4),
            "Time (s)": r.total_elapsed_seconds,
        })

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["Dataset", "Strategy"])
        .agg(
            Baseline_F1=("Baseline Val F1", "mean"),
            Val_F1_Mean=("Best Val F1", "mean"),
            Val_F1_Std=("Best Val F1", "std"),
            Gain_Mean=("Gain (+F1)", "mean"),
            Holdout_F1_Mean=("Holdout Test F1", "mean"),
            Holdout_F1_Std=("Holdout Test F1", "std"),
            Trials_to_95pct=("Trials to 95% Gain", "mean"),
            Convergence_AUC=("Mean Best-So-Far (AUC)", "mean"),
            Time_s=("Time (s)", "mean"),
        )
        .reset_index()
    )
    agg["Val_F1_Std"] = agg["Val_F1_Std"].fillna(0.0)
    agg["Holdout_F1_Std"] = agg["Holdout_F1_Std"].fillna(0.0)

    lines = [
        "# AutoResearcher — Scientific Benchmark Report",
        "",
        "## 1. Executive Summary & Hypothesis Verification",
        "",
        "> **Technical Hypothesis:** Does an autonomous ML agent that selects its next experiment based on rich validation error diagnostics (train-val overfitting gap, per-class F1 breakdown, confusion matrix pairs) discover higher-performing models in fewer trials than Random Search, Grid Search, and black-box Bayesian Optimization (Optuna TPE)?",
        "",
        "### Quantitative Comparison by Dataset",
        "",
        "| Dataset | Strategy | Baseline F1 | Best Val Macro-F1 (Mean ± Std) | Gain over Baseline | Locked Holdout Test F1 | Trials to 95% Peak | Convergence AUC | Mean Time (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in agg.iterrows():
        is_agent = "AutoResearcher" in str(row["Strategy"])
        strat_fmt = f"**{row['Strategy']}**" if is_agent else str(row["Strategy"])
        lines.append(
            f"| `{row['Dataset']}` | {strat_fmt} | {row['Baseline_F1']:.4f} | "
            f"**{row['Val_F1_Mean']:.4f}** ± {row['Val_F1_Std']:.4f} | "
            f"+{row['Gain_Mean']:.4f} | **{row['Holdout_F1_Mean']:.4f}** ± {row['Holdout_F1_Std']:.4f} | "
            f"{row['Trials_to_95pct']:.1f} | {row['Convergence_AUC']:.4f} | {row['Time_s']:.1f}s |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Sample Agent Scientific Lab Notebook (Key Breakthrough Decisions)",
        "",
    ])

    agent_runs = [r for r in results if "AutoResearcher" in r.strategy_name]
    for ar in agent_runs:
        lines.append(f"### Dataset: `{ar.dataset_name}` (Best Val F1: `{ar.best_val_macro_f1:.4f}` | Holdout F1: `{ar.holdout_test_macro_f1:.4f}`)")
        lines.append("")
        for t in ar.trials:
            if t.was_new_best and t.agent_decision is not None:
                lines.append(
                    f"- **Trial #{t.trial_index}** (`Val Macro-F1 = {t.diagnostics.val_macro_f1:.4f}`, "
                    f"`Overfit Gap = {t.diagnostics.overfit_gap:+.4f}`, `Model = {t.config.model_family}`):"
                )
                lines.append(f"  - *Diagnosis:* {t.agent_decision.bottleneck_diagnosis}")
                lines.append(f"  - *Hypothesis:* {t.agent_decision.scientific_hypothesis}")
                lines.append(f"  - *Changes Applied:* `{', '.join(t.agent_decision.changed_components)}`")
        lines.append("")

    report_content = "\n".join(lines)
    output_path.write_text(report_content, encoding="utf-8")
    return report_content


def run_full_benchmark(
    datasets: Optional[List[str]] = None,
    budget: int = 15,
    stochastic_seeds: Optional[List[int]] = None,
    n_splits: int = 5,
    prefer_offline_fallback: bool = False,
    artifacts_dir: Optional[Path] = None,
) -> List[MethodBenchmarkResult]:
    """Run the complete multi-seed benchmark across datasets and generate plots + Markdown report."""
    target_datasets = datasets or DEFAULT_BENCHMARK_DATASETS
    seeds = stochastic_seeds or [42, 101, 202]
    out_dir = Path(artifacts_dir or SETTINGS.artifacts_dir)
    tracker = ExperimentTracker(artifacts_dir=out_dir)

    all_results: List[MethodBenchmarkResult] = []

    for ds_name in target_datasets:
        print(f"\n========================================================")
        print(f" Loading Dataset: {ds_name}")
        print(f"========================================================")
        ds_split = load_benchmark_dataset(
            dataset_name=ds_name,
            split_seed=42,
            prefer_offline_fallback=prefer_offline_fallback,
        )
        print(
            f"Profile: {ds_split.profile.n_samples_total} rows "
            f"({ds_split.profile.n_numeric} num, {ds_split.profile.n_categorical} cat), "
            f"{ds_split.profile.n_classes} classes, Imbalance={ds_split.profile.imbalance_ratio}x"
        )

        # 1. AutoResearcher Agent
        print(f"-> Running AutoResearcher (Agent) [budget={budget}]...")
        res_agent = run_single_strategy(
            strategy_type="agent",
            dataset=ds_split,
            budget=budget,
            seed=seeds[0],
            n_splits=n_splits,
            tracker=tracker,
        )
        all_results.append(res_agent)
        print(
            f"   [Agent] Best Val F1: {res_agent.best_val_macro_f1:.4f} "
            f"(+{res_agent.improvement_over_baseline:.4f}) | Holdout F1: {res_agent.holdout_test_macro_f1:.4f} "
            f"| 95% Peak at Trial #{res_agent.trials_to_95pct_peak}"
        )

        # 2. Optuna (TPE) across seeds
        for s in seeds:
            print(f"-> Running Optuna (TPE) [seed={s}, budget={budget}]...")
            res_opt = run_single_strategy(
                strategy_type="optuna",
                dataset=ds_split,
                budget=budget,
                seed=s,
                n_splits=n_splits,
                tracker=tracker,
            )
            all_results.append(res_opt)
            print(
                f"   [Optuna seed={s}] Best Val F1: {res_opt.best_val_macro_f1:.4f} "
                f"| Holdout F1: {res_opt.holdout_test_macro_f1:.4f}"
            )

        # 3. Random Search across seeds
        for s in seeds:
            print(f"-> Running Random Search [seed={s}, budget={budget}]...")
            res_rnd = run_single_strategy(
                strategy_type="random",
                dataset=ds_split,
                budget=budget,
                seed=s,
                n_splits=n_splits,
                tracker=tracker,
            )
            all_results.append(res_rnd)
            print(
                f"   [Random seed={s}] Best Val F1: {res_rnd.best_val_macro_f1:.4f} "
                f"| Holdout F1: {res_rnd.holdout_test_macro_f1:.4f}"
            )

        # 4. Deterministic Grid Search
        print(f"-> Running Grid Search [budget={budget}]...")
        res_grid = run_single_strategy(
            strategy_type="grid",
            dataset=ds_split,
            budget=budget,
            seed=seeds[0],
            n_splits=n_splits,
            tracker=tracker,
        )
        all_results.append(res_grid)
        print(
            f"   [Grid] Best Val F1: {res_grid.best_val_macro_f1:.4f} "
            f"| Holdout F1: {res_grid.holdout_test_macro_f1:.4f}"
        )

    plots_dir = out_dir / "plots"
    generate_benchmark_plots(all_results, output_dir=plots_dir)
    report_path = out_dir / "BENCHMARK_REPORT.md"
    generate_markdown_report(all_results, output_path=report_path)
    print(f"\nBenchmark complete! Report saved to: {report_path}")
    return all_results
