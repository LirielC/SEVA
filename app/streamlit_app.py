"""Streamlit Dashboard for AutoResearcher: Convergence Curves, Lab Notebook, and Live Runner."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src is importable when launched via `streamlit run app/streamlit_app.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoresearcher.benchmark.runner import generate_markdown_report, run_single_strategy
from autoresearcher.config import DEFAULT_BENCHMARK_DATASETS, SETTINGS
from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.tracking.mlflow_logger import ExperimentTracker
from autoresearcher.tracking.plotting import generate_benchmark_plots

st.set_page_config(
    page_title="SEVA — Secure Experimentation and Validation Agent",
    layout="wide",
)

st.title("SEVA — Secure Experimentation and Validation Agent")
st.markdown(
    """
    **Core Technical Thesis:** *Can an autonomous agent that chooses its next experiment based on rich validation error diagnostics
    (train-vs-val generalization gap, per-class F1, and confusion matrix topologies) converge to superior tabular pipelines in fewer trials
    than Random Search, Grid Search, and Bayesian Optimization (Optuna TPE)?*
    """
)

tracker = ExperimentTracker(artifacts_dir=ROOT_DIR / "artifacts")

with st.sidebar:
    st.header("Execute Live Benchmark")
    selected_ds = st.selectbox("Dataset (OpenML)", options=DEFAULT_BENCHMARK_DATASETS, index=0)
    budget_input = st.slider("Experiment Budget (Trials)", min_value=5, max_value=20, value=12, step=1)
    cv_folds = st.selectbox("Stratified CV Folds", options=[3, 5], index=1)
    seed_input = st.number_input("Random Seed", value=42, step=1)
    offline_toggle = st.checkbox("Fast / Offline Mode", value=False)

    if st.button("Run Benchmark on Dataset", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Loading dataset and isolating locked Holdout Test split (20%)...")
        ds_split = load_benchmark_dataset(
            selected_ds,
            split_seed=int(seed_input),
            prefer_offline_fallback=offline_toggle,
        )
        methods = [("agent", "AutoResearcher (Agent)"), ("optuna", "Optuna (TPE)"), ("random", "Random Search"), ("grid", "Grid Search")]
        total_steps = len(methods) * int(budget_input)
        completed = 0

        for m_code, m_label in methods:
            def _on_step(rec, label=m_label):
                nonlocal completed
                completed += 1
                progress_bar.progress(
                    min(1.0, completed / total_steps),
                    text=f"{label} — Trial {rec.trial_index + 1}/{budget_input} (Best Val F1: {rec.best_val_macro_f1_so_far:.4f})",
                )

            run_single_strategy(
                strategy_type=m_code,
                dataset=ds_split,
                budget=int(budget_input),
                seed=int(seed_input),
                n_splits=int(cv_folds),
                tracker=tracker,
                progress_callback=_on_step,
            )

        all_res = tracker.load_all_results()
        generate_benchmark_plots(all_res, output_dir=ROOT_DIR / "artifacts" / "plots")
        generate_markdown_report(all_res, output_path=ROOT_DIR / "artifacts" / "BENCHMARK_REPORT.md")
        progress_bar.empty()
        st.success(f"Benchmark completed for '{selected_ds}'. Artifacts and figures updated.")

results = tracker.load_all_results()

if not results:
    st.info(
        "No serialized runs found in artifacts/runs. "
        "Click 'Run Benchmark on Dataset' in the sidebar or execute `python -m autoresearcher.cli run-benchmark`."
    )
    st.stop()

available_datasets = sorted({r.dataset_name for r in results})
tab_overview, tab_notebook, tab_configs = st.tabs([
    "Convergence Trajectories & Efficiency",
    "Agent Scientific Lab Notebook",
    "Winning Pipelines & Holdout Evaluation",
])

with tab_overview:
    ds_filter = st.selectbox("Filter Dataset for Trajectory Analysis", options=available_datasets, index=0)
    ds_runs = [r for r in results if r.dataset_name == ds_filter]

    # Top KPIs comparing SEVA vs best baseline on selected dataset
    agent_runs = [r for r in ds_runs if "AutoResearcher" in r.strategy_name]
    other_runs = [r for r in ds_runs if "AutoResearcher" not in r.strategy_name]

    if agent_runs:
        ag = agent_runs[0]
        best_other_holdout = max((r.holdout_test_macro_f1 for r in other_runs), default=ag.baseline_val_macro_f1)
        delta_vs_baselines = ag.holdout_test_macro_f1 - best_other_holdout

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "SEVA — Best CV Macro-F1",
            f"{ag.best_val_macro_f1:.4f}",
            f"+{ag.improvement_over_baseline:.4f} vs Baseline",
        )
        c2.metric(
            "SEVA — Locked Holdout F1",
            f"{ag.holdout_test_macro_f1:.4f}",
            f"{delta_vs_baselines:+.4f} vs Top Baseline",
        )
        c3.metric(
            "Trials to 95% Peak (Efficiency)",
            f"Trial #{ag.trials_to_95pct_peak}",
            f"Budget: {ag.budget} trials",
        )
        c4.metric(
            "Optimal Model Family",
            ag.best_config.model_family.upper(),
            f"FE: {ag.best_config.feature_engineering}",
        )

    # Build Convergence DataFrame
    curve_dict = {}
    for strat in sorted({r.strategy_name for r in ds_runs}):
        s_runs = [r for r in ds_runs if r.strategy_name == strat]
        min_t = min(len(r.trials) for r in s_runs)
        mat = np.array([[t.best_val_macro_f1_so_far for t in r.trials[:min_t]] for r in s_runs])
        curve_dict[strat] = np.mean(mat, axis=0)

    df_curves = pd.DataFrame(curve_dict)
    df_curves.index.name = "Trial"
    st.subheader(f"Convergence Curves (Best Validation Macro-F1 per Trial) — '{ds_filter}'")
    st.line_chart(df_curves, height=390)

    # Summary Table across all datasets
    st.subheader("Global Benchmark Comparison")
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Dataset": r.dataset_name,
            "Strategy": r.strategy_name,
            "Seed": r.seed,
            "Baseline CV F1": r.baseline_val_macro_f1,
            "Best CV Macro-F1": r.best_val_macro_f1,
            "Gain (+F1)": r.improvement_over_baseline,
            "Holdout Test Macro-F1": r.holdout_test_macro_f1,
            "Trials to 95% Gain": r.trials_to_95pct_peak,
            "Time (s)": r.total_elapsed_seconds,
        })
    df_all = pd.DataFrame(summary_rows)
    st.dataframe(df_all.sort_values(["Dataset", "Holdout Test Macro-F1"], ascending=[True, False]), use_container_width=True)

with tab_notebook:
    st.subheader("Scientific Traceability: Error Diagnosis -> Hypothesis -> Intervention")
    agent_only = [r for r in results if "AutoResearcher" in r.strategy_name]
    if not agent_only:
        st.warning("No SEVA agent runs found.")
    else:
        nb_ds = st.selectbox("Lab Notebook Dataset", options=[r.dataset_name for r in agent_only], key="nb_ds")
        selected_run = next(r for r in agent_only if r.dataset_name == nb_ds)

        for t in selected_run.trials:
            badge = "[NEW RECORD]" if t.was_new_best else "[Exploratory]"
            header = (
                f"Trial #{t.trial_index:02d} | {badge} | Val Macro-F1: {t.diagnostics.val_macro_f1:.4f} "
                f"(Best: {t.best_val_macro_f1_so_far:.4f}) | Overfit Gap: {t.diagnostics.overfit_gap:+.4f} "
                f"| Model: {t.config.model_family}"
            )
            with st.expander(header, expanded=t.was_new_best):
                if t.agent_decision:
                    st.markdown(f"**Diagnosis:** {t.agent_decision.bottleneck_diagnosis}")
                    st.markdown(f"**Hypothesis:** {t.agent_decision.scientific_hypothesis}")
                    st.markdown(
                        f"**Mode:** `{t.agent_decision.exploration_mode}` | "
                        f"**Mutations:** `{', '.join(t.agent_decision.changed_components)}`"
                    )
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Per-Class F1 Metrics (5-Fold CV):**")
                    st.json(t.diagnostics.per_class_f1)
                with col_b:
                    st.markdown("**Evaluated PipelineConfig:**")
                    st.json(t.config.model_dump())

with tab_configs:
    st.subheader("Selected Optimal Pipelines Across Strategies")
    for r in results:
        if "AutoResearcher" in r.strategy_name:
            st.markdown(
                f"### {r.dataset_name} — {r.strategy_name} "
                f"(Holdout Macro-F1: `{r.holdout_test_macro_f1:.4f}` | CV Macro-F1: `{r.best_val_macro_f1:.4f}`)"
            )
            st.json(r.best_config.model_dump())
                        f"**🛠️ Modo:** `{t.agent_decision.exploration_mode}` | "
                        f"**Alterações:** `{', '.join(t.agent_decision.changed_components)}`"
                    )
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Métricas por Classe (Validação 5-Fold):**")
                    st.json(t.diagnostics.per_class_f1)
                with col_b:
                    st.markdown("**Configuração Testada (`PipelineConfig`):**")
                    st.json(t.config.model_dump())

with tab_configs:
    st.subheader("Configurações Vencedoras por Estratégia e Dataset")
    for r in results:
        if "AutoResearcher" in r.strategy_name:
            st.markdown(
                f"### 🏆 `{r.dataset_name}` — {r.strategy_name} "
                f"(Holdout Macro-F1: `{r.holdout_test_macro_f1:.4f}` | CV Macro-F1: `{r.best_val_macro_f1:.4f}`)"
            )
            st.json(r.best_config.model_dump())
