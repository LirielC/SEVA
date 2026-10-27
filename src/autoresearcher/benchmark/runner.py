"""Benchmark runner comparing SEVA (LLM Agent & Heuristic Agent) against Optuna (TPE) and Random Search."""

from __future__ import annotations

import json
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
from autoresearcher.schemas import (
    AgentDecision,
    MethodBenchmarkResult,
    PipelineConfig,
    TrialRecord,
)
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots


def _compute_trials_to_95pct_gain(trials: List[TrialRecord]) -> int:
    """Compute earliest trial index that achieved >= 95% of total validation gain over Trial 0."""
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


def _compute_trials_to_fixed_threshold(trials: List[TrialRecord], threshold: float) -> Optional[int]:
    """Compute earliest trial index achieving a pre-defined absolute Macro-F1 threshold."""
    for t in trials:
        if t.best_val_macro_f1_so_far >= threshold - 1e-6:
            return t.trial_index
    return None


def run_single_strategy(
    strategy_type: str,
    dataset: DatasetSplit,
    budget: int = 15,
    seed: int = 42,
    n_splits: int = 5,
    target_threshold: Optional[float] = None,
    tracker: Optional[ExperimentTracker] = None,
    progress_callback: Optional[Callable[[TrialRecord], None]] = None,
) -> MethodBenchmarkResult:
    """Execute one optimization strategy for `budget` trials on a locked dataset split."""
    start_total = time.perf_counter()
    n_classes = dataset.profile.n_classes

    if strategy_type in ("llm_agent", "agent_llm"):
        agent = AutoResearcherAgent(profile=dataset.profile, budget=budget, seed=seed, mode="llm")
        strategy_name = agent.name
    elif strategy_type in ("heuristic_agent", "agent", "agent_heuristic"):
        agent = AutoResearcherAgent(profile=dataset.profile, budget=budget, seed=seed, mode="heuristic")
        strategy_name = agent.name
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

    tot_tokens = 0
    tot_cost = 0.0
    tot_calls = 0

    for trial_idx in range(budget):
        decision: Optional[AgentDecision] = None
        prior_best = best_val_so_far

        if strategy_type in ("llm_agent", "agent_llm", "heuristic_agent", "agent", "agent_heuristic"):
            cfg, decision = agent.suggest_next(history)
            if decision and decision.token_usage:
                tot_tokens += decision.token_usage.total_tokens
                tot_cost += decision.token_usage.estimated_cost_usd
                tot_calls += 1
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

        # Evaluate hypothesis confirmation and formulate observed outcome
        observed_outcome: Optional[str] = None
        hypothesis_confirmed: Optional[bool] = None

        if decision is not None:
            delta_val = round(diag.val_macro_f1 - prior_best, 4) if prior_best > 0 else 0.0
            if trial_idx == 0:
                observed_outcome = (
                    f"Baseline established: Val Macro-F1={diag.val_macro_f1:.4f}, "
                    f"Overfit Gap={diag.overfit_gap:+.4f}, Worst Class F1={diag.worst_class_f1:.4f}."
                )
                hypothesis_confirmed = True
            elif was_new_best:
                observed_outcome = (
                    f"CONFIRMED: New peak reached. Val Macro-F1={diag.val_macro_f1:.4f} ({delta_val:+.4f} vs prior best), "
                    f"Overfit Gap={diag.overfit_gap:+.4f}, Worst Class '{diag.worst_class_label}' F1={diag.worst_class_f1:.4f}."
                )
                hypothesis_confirmed = True
            else:
                observed_outcome = (
                    f"REFUTED/SUB-PEAK: Val Macro-F1={diag.val_macro_f1:.4f} ({delta_val:+.4f} vs peak {prior_best:.4f}), "
                    f"Overfit Gap={diag.overfit_gap:+.4f}."
                )
                hypothesis_confirmed = False

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
            observed_outcome=observed_outcome,
            hypothesis_confirmed=hypothesis_confirmed,
            token_usage=decision.token_usage if decision else None,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

    # Final evaluation ONCE on the locked 20% Holdout Test set
    holdout_f1 = evaluate_on_holdout(best_config, dataset=dataset, seed=seed)
    baseline_f1 = history[0].diagnostics.val_macro_f1
    improvement = round(best_val_so_far - baseline_f1, 5)
    trials_to_95 = _compute_trials_to_95pct_gain(history)
    trials_to_thresh = (
        _compute_trials_to_fixed_threshold(history, target_threshold)
        if target_threshold is not None
        else trials_to_95
    )
    auc_conv = round(float(np.mean([t.best_val_macro_f1_so_far for t in history])), 5)
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
        trials_to_target_threshold=trials_to_thresh,
        auc_convergence=auc_conv,
        total_elapsed_seconds=total_elapsed,
        total_llm_tokens=tot_tokens,
        total_llm_cost_usd=round(tot_cost, 6),
        total_llm_calls=tot_calls,
        best_config=best_config,
        trials=history,
    )

    if tracker is not None:
        tracker.log_method_run(result)

    return result


def replay_pipeline_from_config(
    config_source: str | Path | dict,
    dataset_name: str,
    seed: int = 42,
    n_splits: int = 5,
    prefer_offline_fallback: bool = False,
) -> dict:
    """Reproduce any pipeline run deterministically from its seed and exported JSON configuration."""
    if isinstance(config_source, (str, Path)):
        p = Path(config_source)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            cfg_dict = raw.get("best_config", raw.get("config", raw))
        else:
            cfg_dict = json.loads(str(config_source))
    else:
        cfg_dict = config_source

    config = PipelineConfig.model_validate(cfg_dict)
    ds = load_benchmark_dataset(dataset_name, split_seed=seed, prefer_offline_fallback=prefer_offline_fallback)

    diag, elapsed = evaluate_pipeline_cv(config=config, dataset=ds, n_splits=n_splits, seed=seed)
    holdout_f1 = evaluate_on_holdout(config=config, dataset=ds, seed=seed)

    return {
        "dataset": dataset_name,
        "seed": seed,
        "config_hash": config.config_hash(),
        "cv_val_macro_f1": diag.val_macro_f1,
        "cv_overfit_gap": diag.overfit_gap,
        "cv_worst_class_f1": diag.worst_class_f1,
        "holdout_test_macro_f1": holdout_f1,
        "elapsed_seconds": elapsed,
        "config": config.model_dump(),
    }


def generate_markdown_report(
    results: List[MethodBenchmarkResult],
    output_path: Path,
) -> str:
    """Create a rigorous scientific Markdown report summarizing all benchmark runs and hypotheses."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "Dataset": r.dataset_name,
            "Strategy": r.strategy_name,
            "Seed": r.seed,
            "Baseline Val F1": r.baseline_val_macro_f1,
            "Best Val F1": r.best_val_macro_f1,
            "Gain (+F1)": r.improvement_over_baseline,
            "Holdout Test F1": r.holdout_test_macro_f1,
            "Trials to 95% Gain": r.trials_to_95pct_peak,
            "AUC Convergence": r.auc_convergence,
            "LLM Cost ($)": r.total_llm_cost_usd,
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
            AUC_Conv=("AUC Convergence", "mean"),
            Cost_USD=("LLM Cost ($)", "mean"),
            Time_s=("Time (s)", "mean"),
        )
        .reset_index()
    )
    agg["Val_F1_Std"] = agg["Val_F1_Std"].fillna(0.0)
    agg["Holdout_F1_Std"] = agg["Holdout_F1_Std"].fillna(0.0)

    lines = [
        "# SEVA — Scientific Benchmark Report",
        "",
        "## 1. Executive Summary & Hypothesis Verification",
        "",
        "> **Core Technical Hypothesis:** Does an autonomous agent guided by rich validation error diagnostics (train-val gap, class collapse, confusion matrices) discover superior tabular pipelines with higher sample efficiency and lower regret than blind Random Search and black-box Bayesian Optimization (Optuna TPE)?",
        "",
        "### Four-Strategy Ablation Benchmark",
        "",
        "| Dataset | Strategy | Baseline F1 | Best Val Macro-F1 (Mean ± Std) | Holdout Test F1 (Mean ± Std) | Trials to 95% Peak | AUC Convergence | LLM Cost ($) | Mean Time (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in agg.iterrows():
        is_agent = "SEVA" in str(row["Strategy"]) or "Agent" in str(row["Strategy"])
        strat_fmt = f"**{row['Strategy']}**" if is_agent else str(row["Strategy"])
        cost_str = f"${row['Cost_USD']:.4f}" if row['Cost_USD'] > 0 else "N/A"
        lines.append(
            f"| `{row['Dataset']}` | {strat_fmt} | {row['Baseline_F1']:.4f} | "
            f"**{row['Val_F1_Mean']:.4f}** ± {row['Val_F1_Std']:.4f} | "
            f"**{row['Holdout_F1_Mean']:.4f}** ± {row['Holdout_F1_Std']:.4f} | "
            f"{row['Trials_to_95pct']:.1f} | {row['AUC_Conv']:.4f} | {cost_str} | {row['Time_s']:.1f}s |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Scientific Lab Notebook (Diagnosis -> Hypothesis -> Intervention -> Observed Outcome)",
        "",
    ])

    agent_runs = [r for r in results if "Agent" in r.strategy_name or "SEVA" in r.strategy_name]
    for ar in agent_runs:
        lines.append(f"### Strategy: `{ar.strategy_name}` on Dataset `{ar.dataset_name}` (Seed {ar.seed})")
        lines.append(f"- **Final Performance:** CV Val Macro-F1 = `{ar.best_val_macro_f1:.4f}` | Locked Holdout F1 = `{ar.holdout_test_macro_f1:.4f}`")
        lines.append(f"- **Efficiency:** Peak at Trial `#{ar.trials_to_95pct_peak}` | AUC Convergence = `{ar.auc_convergence:.4f}` | Time = `{ar.total_elapsed_seconds:.1f}s`")
        lines.append("")
        for t in ar.trials:
            if t.was_new_best and t.agent_decision is not None:
                lines.append(
                    f"- **Trial #{t.trial_index}** (`Val F1 = {t.diagnostics.val_macro_f1:.4f}`, "
                    f"`Overfit Gap = {t.diagnostics.overfit_gap:+.4f}`, `Model = {t.config.model_family}`):"
                )
                lines.append(f"  - *Diagnosis:* {t.agent_decision.bottleneck_diagnosis}")
                lines.append(f"  - *Hypothesis:* {t.agent_decision.scientific_hypothesis}")
                lines.append(f"  - *Expected Outcome:* {t.agent_decision.expected_outcome}")
                lines.append(f"  - *Mutations Applied:* `{', '.join(t.agent_decision.changed_components)}`")
                if t.observed_outcome:
                    lines.append(f"  - *Observed Outcome:* {t.observed_outcome}")
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
    include_llm_agent: bool = True,
    artifacts_dir: Optional[Path] = None,
) -> List[MethodBenchmarkResult]:
    """Run the 4-way ablation benchmark (Random, Optuna, Heuristic Agent, LLM Agent) across seeds."""
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

        # 1. SEVA Heuristic Diagnostic Agent
        for s in seeds:
            print(f"-> Running SEVA (Heuristic Agent) [seed={s}, budget={budget}]...")
            res_heur = run_single_strategy(
                strategy_type="heuristic_agent",
                dataset=ds_split,
                budget=budget,
                seed=s,
                n_splits=n_splits,
                tracker=tracker,
            )
            all_results.append(res_heur)
            print(
                f"   [Heuristic Agent seed={s}] Best Val F1: {res_heur.best_val_macro_f1:.4f} "
                f"| Holdout F1: {res_heur.holdout_test_macro_f1:.4f} | 95% Peak: Trial #{res_heur.trials_to_95pct_peak}"
            )

        # 2. SEVA LLM Agent (if requested and configured)
        if include_llm_agent:
            for s in seeds:
                print(f"-> Running SEVA (LLM Agent) [seed={s}, budget={budget}]...")
                res_llm = run_single_strategy(
                    strategy_type="llm_agent",
                    dataset=ds_split,
                    budget=budget,
                    seed=s,
                    n_splits=n_splits,
                    tracker=tracker,
                )
                all_results.append(res_llm)
                print(
                    f"   [LLM Agent seed={s}] Best Val F1: {res_llm.best_val_macro_f1:.4f} "
                    f"| Holdout F1: {res_llm.holdout_test_macro_f1:.4f} | Cost: ${res_llm.total_llm_cost_usd:.4f}"
                )

        # 3. Optuna (TPE)
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

        # 4. Uniform Random Search
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

    plots_dir = out_dir / "plots"
    generate_benchmark_plots(all_results, output_dir=plots_dir)
    report_path = out_dir / "BENCHMARK_REPORT.md"
    generate_markdown_report(all_results, output_path=report_path)
    print(f"\nBenchmark complete! Report saved to: {report_path}")
    return all_results
