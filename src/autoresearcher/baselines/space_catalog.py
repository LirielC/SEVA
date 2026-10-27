"""Shared declarative catalog of valid values for the 5-layer CASH + Pipeline space."""

from __future__ import annotations

from typing import Dict, List, Set
import numpy as np

from autoresearcher.schemas import PipelineConfig

SEARCH_SPACE_CATALOG: Dict[str, List[object]] = {
    "imputer": ["median", "mean"],
    "scaler": ["none", "standard", "robust", "minmax"],
    "categorical_encoder": ["onehot", "ordinal", "target_smooth"],
    "feature_engineering": [
        "none",
        "poly_top5",
        "ratios_top5",
        "mutual_info_k80",
        "tree_importance_k75",
    ],
    "imbalance_strategy": [
        "none",
        "class_weight_balanced",
        "random_oversample",
        "random_undersample",
    ],
    "threshold_tuning": [False, True],
    "model_family": ["random_forest", "xgboost", "catboost"],
    "n_estimators": [80, 120, 180, 250, 350],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.08, 0.12, 0.20],
    "regularization": [0.1, 0.5, 1.0, 3.0, 6.0, 12.0],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "min_samples_leaf": [1, 2, 4, 8, 12],
}


def sample_random_config(
    rng: np.random.Generator,
    seen_hashes: Set[str],
    n_classes: int = 2,
) -> PipelineConfig:
    """Sample a unique valid PipelineConfig uniformly from the shared search space."""
    for _ in range(250):
        th_choices = SEARCH_SPACE_CATALOG["threshold_tuning"] if n_classes == 2 else [False]
        cfg = PipelineConfig(
            imputer=str(rng.choice(SEARCH_SPACE_CATALOG["imputer"])),  # type: ignore[arg-type]
            scaler=str(rng.choice(SEARCH_SPACE_CATALOG["scaler"])),  # type: ignore[arg-type]
            categorical_encoder=str(rng.choice(SEARCH_SPACE_CATALOG["categorical_encoder"])),  # type: ignore[arg-type]
            feature_engineering=str(rng.choice(SEARCH_SPACE_CATALOG["feature_engineering"])),  # type: ignore[arg-type]
            imbalance_strategy=str(rng.choice(SEARCH_SPACE_CATALOG["imbalance_strategy"])),  # type: ignore[arg-type]
            threshold_tuning=bool(rng.choice(th_choices)),
            model_family=str(rng.choice(SEARCH_SPACE_CATALOG["model_family"])),  # type: ignore[arg-type]
            n_estimators=int(rng.choice(SEARCH_SPACE_CATALOG["n_estimators"])),
            max_depth=int(rng.choice(SEARCH_SPACE_CATALOG["max_depth"])),
            learning_rate=float(rng.choice(SEARCH_SPACE_CATALOG["learning_rate"])),
            regularization=float(rng.choice(SEARCH_SPACE_CATALOG["regularization"])),
            subsample=float(rng.choice(SEARCH_SPACE_CATALOG["subsample"])),
            min_samples_leaf=int(rng.choice(SEARCH_SPACE_CATALOG["min_samples_leaf"])),
        )
        if cfg.config_hash() not in seen_hashes:
            return cfg
    return PipelineConfig.default_baseline()
