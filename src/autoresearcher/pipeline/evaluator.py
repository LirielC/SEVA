"""Cross-validation executor, threshold optimizer, and rich ErrorAnalyzer."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple
import warnings
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from autoresearcher.data.openml_loader import DatasetSplit
from autoresearcher.pipeline.builder import (
    apply_fold_resampling,
    build_preprocessor_and_fe,
    fit_estimator_with_optional_weights,
    instantiate_classifier,
)
from autoresearcher.schemas import ConfusedPair, DiagnosticReport, PipelineConfig


def _find_best_binary_threshold(y_true: np.ndarray, pos_probs: np.ndarray) -> float:
    """Find probability threshold in [0.20, 0.80] that maximizes Macro-F1 on training/inner fold."""
    if len(pos_probs) == 0:
        return 0.50

    probs = np.asarray(pos_probs, dtype=float)
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        warnings.warn(
            "pos_probs contains NaN or Inf values; degenerate model predictions cannot be optimized. "
            "Defaulting safely to neutral threshold 0.50 without optimization.",
            UserWarning,
            stacklevel=2,
        )
        return 0.50

    if np.ptp(probs) < 1e-5:
        return 0.50

    y_arr = np.nan_to_num(np.asarray(y_true, dtype=int), nan=0)
    y_true_mean = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.50

    thresholds = np.linspace(0.22, 0.78, 15)
    scores = []
    for th in thresholds:
        preds = (probs >= th).astype(int)
        score = float(f1_score(y_arr, preds, average="macro", zero_division=0))
        scores.append(score)

    max_score = max(scores)
    # Collect candidate thresholds within 1e-5 of the true global maximum score
    candidates = [
        float(th) for th, sc in zip(thresholds, scores)
        if sc >= max_score - 1e-5
    ]

    # Primary: threshold closest to 0.50
    # Secondary: for equidistant pairs around 0.50 (e.g. 0.46 vs 0.54), select threshold
    # that minimizes difference between predicted positive rate and training set prevalence
    def tie_break_key(th: float) -> Tuple[float, float]:
        dist_to_half = round(abs(th - 0.50), 6)
        preds = (probs >= th).astype(int)
        pred_rate = float(np.mean(preds))
        bias = abs(pred_rate - y_true_mean)
        return (dist_to_half, bias)

    best_th = min(candidates, key=tie_break_key)
    return float(best_th)


def evaluate_pipeline_cv(
    config: PipelineConfig,
    dataset: DatasetSplit,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[DiagnosticReport, float]:
    """Run leak-free Stratified K-Fold CV on Dev set and produce a rich DiagnosticReport."""
    start_t = time.perf_counter()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_dev = dataset.X_dev
    y_dev = dataset.y_dev
    n_classes = dataset.profile.n_classes
    class_names = dataset.class_names

    oof_preds = np.zeros_like(y_dev)
    fold_val_f1s: List[float] = []
    fold_train_f1s: List[float] = []
    feature_importances_accum: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        fold_seed = seed + fold_idx * 17
        X_tr_raw = X_dev.iloc[train_idx].reset_index(drop=True)
        y_tr_raw = y_dev[train_idx]
        X_va_raw = X_dev.iloc[val_idx].reset_index(drop=True)
        y_va_raw = y_dev[val_idx]

        # Fit preprocessing + feature engineering strictly on training fold
        preproc = build_preprocessor_and_fe(config, X_tr_raw, random_state=fold_seed)
        X_tr_trans = preproc.fit_transform(X_tr_raw, y_tr_raw)
        X_va_trans = preproc.transform(X_va_raw)

        # Apply training fold resampling (Layer 4)
        X_tr_bal, y_tr_bal = apply_fold_resampling(
            X_tr_trans,
            y_tr_raw,
            strategy=config.imbalance_strategy,
            random_state=fold_seed,
        )

        clf = instantiate_classifier(config, n_classes=n_classes, random_state=fold_seed)
        fit_estimator_with_optional_weights(clf, X_tr_bal, y_tr_bal, config)

        # Predict on training fold (un-resampled to measure true train-val gap) and validation fold
        if config.threshold_tuning and n_classes == 2 and hasattr(clf, "predict_proba"):
            tr_probs = clf.predict_proba(X_tr_trans)[:, 1]
            opt_th = _find_best_binary_threshold(y_tr_raw, tr_probs)
            tr_preds = (tr_probs >= opt_th).astype(int)
            va_probs = clf.predict_proba(X_va_trans)[:, 1]
            va_preds = (va_probs >= opt_th).astype(int)
        else:
            tr_preds = np.asarray(clf.predict(X_tr_trans)).reshape(-1).astype(int)
            va_preds = np.asarray(clf.predict(X_va_trans)).reshape(-1).astype(int)

        oof_preds[val_idx] = va_preds
        fold_train_f1s.append(float(f1_score(y_tr_raw, tr_preds, average="macro", zero_division=0)))
        fold_val_f1s.append(float(f1_score(y_va_raw, va_preds, average="macro", zero_division=0)))

        if hasattr(clf, "feature_importances_"):
            fi = np.asarray(clf.feature_importances_, dtype=float)
            if fi.ndim == 1 and len(fi) > 0:
                feature_importances_accum.append(fi)

    val_macro_f1 = round(float(np.mean(fold_val_f1s)), 5)
    val_std_f1 = round(float(np.std(fold_val_f1s)), 5)
    train_macro_f1 = round(float(np.mean(fold_train_f1s)), 5)
    overfit_gap = round(float(train_macro_f1 - val_macro_f1), 5)
    val_bal_acc = round(float(balanced_accuracy_score(y_dev, oof_preds)), 5)

    # Per-class F1 breakdown
    raw_per_class = f1_score(y_dev, oof_preds, average=None, zero_division=0)
    per_class_f1: Dict[str, float] = {
        class_names[i]: round(float(raw_per_class[i]), 4) for i in range(len(class_names))
    }
    sorted_classes = sorted(per_class_f1.items(), key=lambda kv: kv[1])
    worst_class_label, worst_class_f1 = sorted_classes[0]
    best_class_f1 = sorted_classes[-1][1]

    # Confusion Matrix analysis: extract top off-diagonal confused pairs
    cm = confusion_matrix(y_dev, oof_preds, labels=list(range(n_classes)))
    confused_pairs: List[ConfusedPair] = []
    for i in range(n_classes):
        row_total = max(1, int(cm[i].sum()))
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append(
                    ConfusedPair(
                        true_class=class_names[i],
                        predicted_class=class_names[j],
                        error_count=int(cm[i, j]),
                        error_rate_within_true_class=round(float(cm[i, j]) / row_total, 4),
                    )
                )
    confused_pairs.sort(key=lambda cp: cp.error_count, reverse=True)
    top_confused = confused_pairs[:3]

    # Top feature importance indices
    top_features: List[str] = []
    if feature_importances_accum and all(len(x) == len(feature_importances_accum[0]) for x in feature_importances_accum):
        mean_fi = np.mean(np.vstack(feature_importances_accum), axis=0)
        n_fi = len(mean_fi)
        top_idx = np.argsort(mean_fi)[::-1][:5]
        try:
            feat_names = list(preproc.get_feature_names_out())
        except Exception:
            feat_names = None

        for idx in top_idx:
            # Strictly verify that feature names length matches estimator dimensionality
            if feat_names is not None and len(feat_names) == n_fi and idx < len(feat_names):
                col_label = str(feat_names[idx])
            else:
                col_label = f"feature_{idx}"
            top_features.append(f"{col_label} ({mean_fi[idx]:.3f})")

    # Classify Primary Pathology for explicit diagnostic guidance
    class_gap = best_class_f1 - worst_class_f1
    if class_gap >= 0.22 and dataset.profile.imbalance_ratio >= 1.5 and config.imbalance_strategy == "none":
        primary_pathology = "minority_class_collapse"
    elif overfit_gap >= 0.14:
        primary_pathology = "severe_overfitting"
    elif train_macro_f1 < 0.72 and val_macro_f1 < 0.68:
        primary_pathology = "underfitting_low_capacity"
    elif config.feature_engineering == "none" and val_macro_f1 < 0.80:
        primary_pathology = "feature_noise_or_representation"
    else:
        primary_pathology = "well_balanced_needs_fine_tuning"

    top_err_str = (
        f"{top_confused[0].true_class}->{top_confused[0].predicted_class} ({top_confused[0].error_count} errors)"
        if top_confused
        else "none"
    )
    summary_text = (
        f"Val Macro-F1={val_macro_f1:.4f} (±{val_std_f1:.4f}), Train Macro-F1={train_macro_f1:.4f} "
        f"(Overfit Gap={overfit_gap:+.4f}). Worst Class='{worst_class_label}' (F1={worst_class_f1:.4f} vs Best={best_class_f1:.4f}). "
        f"Top Confusion: {top_err_str}. Primary Pathology: {primary_pathology}."
    )

    elapsed = round(time.perf_counter() - start_t, 4)
    report = DiagnosticReport(
        val_macro_f1=val_macro_f1,
        val_std_f1=val_std_f1,
        train_macro_f1=train_macro_f1,
        overfit_gap=overfit_gap,
        val_balanced_accuracy=val_bal_acc,
        per_class_f1=per_class_f1,
        worst_class_label=worst_class_label,
        worst_class_f1=worst_class_f1,
        best_class_f1=best_class_f1,
        top_confused_pairs=top_confused,
        top_features=top_features,
        primary_pathology=primary_pathology,
        diagnostic_summary_text=summary_text,
    )
    return report, elapsed


def evaluate_on_holdout(
    config: PipelineConfig,
    dataset: DatasetSplit,
    seed: int = 42,
) -> float:
    """Fit the final selected pipeline on 100% of Dev data and evaluate ONCE on locked Holdout Test set."""
    X_dev = dataset.X_dev
    y_dev = dataset.y_dev
    X_test = dataset.X_test
    y_test = dataset.y_test
    n_classes = dataset.profile.n_classes

    preproc = build_preprocessor_and_fe(config, X_dev, random_state=seed)
    X_dev_trans = preproc.fit_transform(X_dev, y_dev)
    X_test_trans = preproc.transform(X_test)

    X_dev_bal, y_dev_bal = apply_fold_resampling(
        X_dev_trans,
        y_dev,
        strategy=config.imbalance_strategy,
        random_state=seed,
    )

    clf = instantiate_classifier(config, n_classes=n_classes, random_state=seed)
    fit_estimator_with_optional_weights(clf, X_dev_bal, y_dev_bal, config)

    if config.threshold_tuning and n_classes == 2 and hasattr(clf, "predict_proba"):
        dev_probs = clf.predict_proba(X_dev_trans)[:, 1]
        opt_th = _find_best_binary_threshold(y_dev, dev_probs)
        test_probs = clf.predict_proba(X_test_trans)[:, 1]
        test_preds = (test_probs >= opt_th).astype(int)
    else:
        test_preds = np.asarray(clf.predict(X_test_trans)).reshape(-1).astype(int)

    return round(float(f1_score(y_test, test_preds, average="macro", zero_division=0)), 5)
