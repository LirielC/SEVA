"""Generate publication-quality convergence and sample-efficiency charts for README and Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoresearcher.schemas import MethodBenchmarkResult

STRATEGY_COLORS: Dict[str, str] = {
    "SEVA (LLM Agent)": "#8B5CF6",        # Vibrant Violet
    "SEVA (Heuristic Agent)": "#10B981",  # Emerald Green
    "AutoResearcher (Agent)": "#10B981",  # Emerald Green
    "Optuna (TPE)": "#3B82F6",            # Royal Blue
    "Random Search": "#F59E0B",           # Amber
    "Grid Search": "#6B7280",             # Slate Gray
}


def generate_benchmark_plots(
    results: List[MethodBenchmarkResult],
    output_dir: Path,
) -> List[Path]:
    """Generate convergence curves per dataset and an overall Holdout & Sample Efficiency comparison chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []
    if not results:
        return generated_files

    datasets = sorted({r.dataset_name for r in results})

    # 1. Combined Convergence Curve subplot across all datasets
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), squeeze=False)
    for idx, ds_name in enumerate(datasets):
        ax = axes[0, idx]
        ds_runs = [r for r in results if r.dataset_name == ds_name]
        strategies = sorted({r.strategy_name for r in ds_runs})

        for strat in strategies:
            strat_runs = [r for r in ds_runs if r.strategy_name == strat]
            curves = []
            for r in strat_runs:
                curve = [t.best_val_macro_f1_so_far for t in sorted(r.trials, key=lambda x: x.trial_index)]
                curves.append(curve)
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            mat = np.array([c[:min_len] for c in curves], dtype=float)
            mean_curve = np.mean(mat, axis=0)
            std_curve = np.std(mat, axis=0)
            x_steps = np.arange(min_len)

            color = STRATEGY_COLORS.get(strat, "#8B5CF6")
            lw = 2.8 if "AutoResearcher" in strat else 1.8
            ax.plot(x_steps, mean_curve, label=strat, color=color, linewidth=lw, marker="o", markersize=4)
            if len(curves) > 1:
                ax.fill_between(
                    x_steps,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color,
                    alpha=0.15,
                )

        ax.set_title(f"Convergence on '{ds_name}' (5-Fold CV)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial Budget Step (0 = Baseline)", fontsize=10)
        ax.set_ylabel("Best Validation Macro-F1 so far", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    conv_path = output_dir / "convergence_curves.png"
    fig.savefig(conv_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(conv_path)

    # 2. Holdout Test Macro-F1 & Sample Efficiency Bar Plot
    rows = []
    for r in results:
        rows.append({
            "Dataset": r.dataset_name,
            "Strategy": r.strategy_name,
            "Holdout Macro-F1": r.holdout_test_macro_f1,
            "Trials to 95% Peak": r.trials_to_95pct_peak,
        })
    df_sum = pd.DataFrame(rows)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    piv_f1 = df_sum.groupby(["Dataset", "Strategy"])["Holdout Macro-F1"].mean().unstack()
    piv_f1.plot(
        kind="bar",
        ax=ax1,
        color=[STRATEGY_COLORS.get(c, "#8B5CF6") for c in piv_f1.columns],
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.set_title("Locked Holdout Test Macro-F1 (Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Holdout Test Macro-F1")
    ax1.set_ylim(bottom=max(0.4, float(df_sum["Holdout Macro-F1"].min()) - 0.08))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.tick_params(axis="x", rotation=0)

    piv_eff = df_sum.groupby(["Dataset", "Strategy"])["Trials to 95% Peak"].mean().unstack()
    piv_eff.plot(
        kind="bar",
        ax=ax2,
        color=[STRATEGY_COLORS.get(c, "#8B5CF6") for c in piv_eff.columns],
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_title("Sample Efficiency: Trials Needed to Reach 95% of Peak Gain (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Trials")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    summary_plot_path = output_dir / "holdout_and_efficiency.png"
    fig2.savefig(summary_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    generated_files.append(summary_plot_path)

    return generated_files
