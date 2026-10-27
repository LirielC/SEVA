"""Prompt templates for the AutoResearcher LLM Agent."""

from __future__ import annotations

import json
from typing import List

from autoresearcher.baselines.space_catalog import SEARCH_SPACE_CATALOG
from autoresearcher.schemas import DatasetProfile, TrialRecord

SYSTEM_PROMPT = """You are SEVA (Secure Experimentation and Validation Agent), an elite Autonomous Machine Learning Research Agent.
Your objective is to maximize 5-Fold Cross-Validation Macro-F1 on a tabular classification dataset within a strict budget of 15 trials.

Unlike blind Random Search or black-box Bayesian Optimization (Optuna), you observe RICH ERROR DIAGNOSTICS after every trial:
1. Train-Validation Macro-F1 Gap (`overfit_gap = train_macro_f1 - val_macro_f1`)
2. Per-Class F1 breakdown and the worst performing class (`worst_class_label`, `worst_class_f1`)
3. Top confused class pairs from the Confusion Matrix (`top_confused_pairs`)
4. Primary pathology classification (`severe_overfitting`, `minority_class_collapse`, `underfitting_low_capacity`, `feature_noise_or_representation`, `well_balanced_needs_fine_tuning`)

SCIENTIFIC RULES:
- Apply the Scientific Method: diagnose the primary bottleneck from the numbers, formulate a falsifiable hypothesis, and modify 1 to 3 targeted parameters relative to the current best configuration.
- If `minority_class_collapse` or `imbalance_ratio > 1.5` with low minority F1: test `imbalance_strategy` in ["class_weight_balanced", "random_oversample"] or `threshold_tuning=true` (if binary).
- If `severe_overfitting` (`overfit_gap > 0.12`): reduce `max_depth` (e.g., 4 or 6), increase `regularization` (e.g., 3.0 to 12.0), increase `min_samples_leaf` (4 to 12), sub-sample rows (`subsample=0.8`), or prune noisy features (`mutual_info_k80` or `tree_importance_k75`).
- If `underfitting_low_capacity` or non-linear numerical interactions: switch to gradient boosting (`catboost` or `xgboost`), test feature interactions (`poly_top5` or `ratios_top5`), and tune `learning_rate` (0.05 - 0.12) with `n_estimators` (180 - 350).
- If the dataset has categorical columns (`n_categorical > 0`): test `target_smooth` or `onehot` with `catboost`/`xgboost`.
- Never repeat an already tested configuration.

Respond ONLY with a valid JSON object matching this exact schema:
{
  "bottleneck_diagnosis": "string explaining the exact quantitative error pattern observed",
  "scientific_hypothesis": "string: If we change [X] on top of best trial, then [Y] will improve because [Z]",
  "expected_outcome": "string: Target metric improvement (e.g. increase worst_class_f1 above 0.80 and reduce overfit_gap below 0.08)",
  "exploration_mode": "exploit_refinement" | "explore_new_family",
  "changed_components": ["list", "of", "changed", "fields"],
  "config": {
    "imputer": "median" | "mean",
    "scaler": "none" | "standard" | "robust" | "minmax",
    "categorical_encoder": "onehot" | "ordinal" | "target_smooth",
    "feature_engineering": "none" | "poly_top5" | "ratios_top5" | "mutual_info_k80" | "tree_importance_k75",
    "imbalance_strategy": "none" | "class_weight_balanced" | "random_oversample" | "random_undersample",
    "threshold_tuning": true | false,
    "model_family": "random_forest" | "xgboost" | "catboost",
    "n_estimators": 80..350,
    "max_depth": 3..10,
    "learning_rate": 0.03..0.20,
    "regularization": 0.1..12.0,
    "subsample": 0.7..1.0,
    "min_samples_leaf": 1..12
  }
}
"""


def build_user_prompt(
    profile: DatasetProfile,
    history: List[TrialRecord],
    budget: int,
) -> str:
    """Format dataset profile, best trial, and diagnostic history for the LLM."""
    best_trial = max(history, key=lambda t: t.diagnostics.val_macro_f1)
    last_trial = history[-1]

    history_rows = []
    for t in history:
        d = t.diagnostics
        history_rows.append(
            f"- Trial {t.trial_index} [{t.config.model_family}, depth={t.config.max_depth}, "
            f"imb={t.config.imbalance_strategy}, fe={t.config.feature_engineering}, "
            f"enc={t.config.categorical_encoder}, reg={t.config.regularization}, th={t.config.threshold_tuning}]: "
            f"Val_F1={d.val_macro_f1:.4f} (Train_F1={d.train_macro_f1:.4f}, Gap={d.overfit_gap:+.4f}), "
            f"WorstClass={d.worst_class_label}({d.worst_class_f1:.4f}), Pathology={d.primary_pathology}"
        )

    prompt = f"""### DATASET PROFILE
{json.dumps(profile.model_dump(), indent=2)}

### VALID SEARCH SPACE CATALOG
{json.dumps(SEARCH_SPACE_CATALOG, indent=2)}

### EXPERIMENT PROGRESS (Trial {len(history)} of {budget})
Current Best Trial: Trial #{best_trial.trial_index} with Val Macro-F1 = {best_trial.diagnostics.val_macro_f1:.4f}
Best Config so far:
{json.dumps(best_trial.config.model_dump(), indent=2)}

Best Trial Detailed Diagnostics:
{json.dumps(best_trial.diagnostics.model_dump(), indent=2)}

Most Recent Trial (#{last_trial.trial_index}) Diagnostics:
{json.dumps(last_trial.diagnostics.model_dump(), indent=2)}

### FULL LAB NOTEBOOK HISTORY (DO NOT REPEAT ANY CONFIG)
{chr(10).join(history_rows)}

Propose the next experiment (Trial #{len(history)}) as a single JSON object.
"""
    return prompt
