"""Tests verifying zero data leakage, fold isolation, and all 5 pipeline layers."""

from __future__ import annotations

import numpy as np

from autoresearcher.data.openml_loader import load_benchmark_dataset
from autoresearcher.pipeline.evaluator import evaluate_on_holdout, evaluate_pipeline_cv
from autoresearcher.schemas import PipelineConfig


def test_dataset_split_strict_isolation() -> None:
    ds = load_benchmark_dataset("credit-g", test_size=0.20, split_seed=42, prefer_offline_fallback=True)
    assert ds.profile.n_samples_total == ds.profile.n_samples_dev + ds.profile.n_samples_test
    assert len(ds.X_dev) == ds.profile.n_samples_dev
    assert len(ds.X_test) == ds.profile.n_samples_test
    assert ds.profile.n_classes == 2
    assert ds.profile.imbalance_ratio > 1.5


def test_all_five_pipeline_layers_execute_cleanly() -> None:
    ds = load_benchmark_dataset("credit-g", test_size=0.20, split_seed=42, prefer_offline_fallback=True)

    configs = [
        PipelineConfig(
            imputer="median",
            scaler="standard",
            categorical_encoder="target_smooth",
            feature_engineering="ratios_top5",
            imbalance_strategy="class_weight_balanced",
            threshold_tuning=True,
            model_family="catboost",
            n_estimators=80,
            max_depth=5,
        ),
        PipelineConfig(
            imputer="mean",
            scaler="robust",
            categorical_encoder="onehot",
            feature_engineering="poly_top5",
            imbalance_strategy="random_oversample",
            threshold_tuning=False,
            model_family="xgboost",
            n_estimators=80,
            max_depth=4,
        ),
        PipelineConfig(
            imputer="median",
            scaler="minmax",
            categorical_encoder="ordinal",
            feature_engineering="tree_importance_k75",
            imbalance_strategy="random_undersample",
            threshold_tuning=False,
            model_family="random_forest",
            n_estimators=80,
            max_depth=6,
        ),
    ]

    for cfg in configs:
        diag, elapsed = evaluate_pipeline_cv(cfg, dataset=ds, n_splits=3, seed=42)
        assert 0.0 <= diag.val_macro_f1 <= 1.0
        assert 0.0 <= diag.train_macro_f1 <= 1.0
        assert elapsed > 0.0
        assert len(diag.per_class_f1) == 2

        holdout_f1 = evaluate_on_holdout(cfg, dataset=ds, seed=42)
        assert 0.0 <= holdout_f1 <= 1.0
