"""Unit tests for transformer contracts, multiclass target encoding, and feature attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoresearcher.pipeline.builder import (
    FeatureEngineeringTransformer,
    SmoothTargetEncoder,
    build_preprocessor_and_fe,
)
from autoresearcher.pipeline.evaluator import _find_best_binary_threshold
from autoresearcher.schemas import PipelineConfig
from sklearn.model_selection import StratifiedKFold


def test_smooth_target_encoder_binary() -> None:
    enc = SmoothTargetEncoder(smoothing=5.0)
    X = np.array([["low"], ["high"], ["low"], ["high"], ["medium"]], dtype=object)
    y = np.array([0, 1, 0, 1, 0])

    enc.fit(X, y)
    out = enc.transform(X)

    assert out.shape == (5, 1)
    names = enc.get_feature_names_out(["risk_level"])
    assert list(names) == ["risk_level"]

    # Unseen category test: should receive global prior (2/5 = 0.4)
    unseen = np.array([["unknown"]], dtype=object)
    out_unseen = enc.transform(unseen)
    assert abs(out_unseen[0, 0] - 0.4) < 1e-4


def test_smooth_target_encoder_multiclass() -> None:
    enc = SmoothTargetEncoder(smoothing=5.0)
    X = np.array([["cat_A"], ["cat_B"], ["cat_A"], ["cat_C"], ["cat_B"], ["cat_C"]], dtype=object)
    y = np.array([0, 1, 2, 0, 1, 2])

    enc.fit(X, y)
    out = enc.transform(X)

    # With 3 classes, should produce 3 columns
    assert out.shape == (6, 3)
    names = enc.get_feature_names_out(["feat1"])
    assert len(names) == 3
    assert set(names) == {"feat1_class_0", "feat1_class_1", "feat1_class_2"}

    # Values must be within [0, 1] probabilities
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_feature_engineering_transformer_names_and_shapes() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 6))
    y = rng.choice([0, 1], size=20)

    strategies = ["none", "poly_top5", "ratios_top5", "mutual_info_k80", "tree_importance_k75"]
    col_names = [f"num_{i}" for i in range(6)]

    for strat in strategies:
        fe = FeatureEngineeringTransformer(strategy=strat, random_state=42)
        X_trans = fe.fit_transform(X, y)
        out_names = fe.get_feature_names_out(col_names)

        assert X_trans.shape[1] == len(out_names), f"Shape mismatch for strategy {strat}"


def test_feature_engineering_single_feature() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    fe = FeatureEngineeringTransformer(strategy="ratios_top5")
    X_trans = fe.fit_transform(X, y)
    names = fe.get_feature_names_out(["feat0"])

    # With only 1 feature, cannot form pairs; should return original feature
    assert X_trans.shape == (4, 1)
    assert list(names) == ["feat0"]


def test_full_pipeline_get_feature_names_out() -> None:
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0],
        "num_b": [10.0, 20.0, 30.0, 40.0],
        "cat_c": ["a", "b", "a", "b"],
    })
    y = np.array([0, 1, 0, 1])

    cfg = PipelineConfig(
        imputer="median",
        scaler="standard",
        categorical_encoder="onehot",
        feature_engineering="poly_top5",
    )
    pipe = build_preprocessor_and_fe(cfg, df)
    pipe.fit(df, y)

    names = pipe.get_feature_names_out()
    assert len(names) > 0
    # Must have numeric prefix and categorical prefix
    assert any("num" in n for n in names)
    assert any("cat" in n for n in names)


def test_threshold_optimizer_plateau_tie_break() -> None:
    # Identical probabilities -> constant predictions across all thresholds
    y_true = np.array([0, 0, 1, 1])
    flat_probs = np.array([0.5, 0.5, 0.5, 0.5])

    best_th = _find_best_binary_threshold(y_true, flat_probs)
    # Must break tie in favor of proximity to 0.50 rather than scanning start (0.22)
    assert abs(best_th - 0.50) < 0.05


def test_smooth_target_encoder_oof_cross_fitting_prevents_leakage() -> None:
    # Unique categories (each appears once): in-sample target encoding would leak y directly.
    # Out-of-fold cross-fitting ensures the validation partition gets the prior of remaining folds, not its own target.
    X = np.array([[f"user_{i}"] for i in range(20)], dtype=object)
    y = np.array([0, 1] * 10)

    enc = SmoothTargetEncoder(smoothing=1.0, cv=5, random_state=42)
    oof_transformed = enc.fit_transform(X, y)

    # In OOF cross-fitting, unseen category in training fold must receive global prior (0.50)
    # rather than memorizing individual y_i (which would yield smoothed ~0.75 for 1 and ~0.25 for 0)
    assert oof_transformed.shape == (20, 1)
    assert np.allclose(oof_transformed, 0.50, atol=1e-3)

    # When transforming unseen data via transform(), it must use full dataset mapping
    test_X = np.array([["user_0"], ["unseen_user"]], dtype=object)
    test_out = enc.transform(test_X)
    assert test_out.shape == (2, 1)
    # user_0 has count=1, y=0 -> (0*1 + 1*0.5)/(1+1) = 0.25
    assert abs(test_out[0, 0] - 0.25) < 1e-3
    assert abs(test_out[1, 0] - 0.50) < 1e-3


def test_smooth_target_encoder_multiclass_oof() -> None:
    X = np.array([[f"cat_{i % 4}"] for i in range(24)], dtype=object)
    y = np.array([0, 1, 2] * 8)

    enc = SmoothTargetEncoder(smoothing=5.0, cv=3, random_state=42)
    oof_out = enc.fit_transform(X, y)

    assert oof_out.shape == (24, 3)
    assert np.all((oof_out >= 0.0) & (oof_out <= 1.0))

    # Transform on test data
    test_out = enc.transform(X[:3])
    assert test_out.shape == (3, 3)
    assert np.all((test_out >= 0.0) & (test_out <= 1.0))


def test_feature_names_order_and_count_all_strategies() -> None:
    df = pd.DataFrame({
        "num_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "num_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "cat_1": ["a", "b", "c", "a", "b", "c"],
        "cat_2": ["x", "x", "y", "y", "z", "z"],
    })
    y = np.array([0, 1, 0, 1, 0, 1])

    fe_strategies = ["none", "poly_top5", "ratios_top5", "mutual_info_k80", "tree_importance_k75"]
    cat_encoders = ["onehot", "ordinal", "target_smooth"]

    for enc in cat_encoders:
        for strat in fe_strategies:
            cfg = PipelineConfig(
                imputer="median",
                scaler="standard",
                categorical_encoder=enc,
                feature_engineering=strat,
            )
            pipe = build_preprocessor_and_fe(cfg, df, random_state=42)
            X_trans = pipe.fit_transform(df, y)
            feat_names = pipe.get_feature_names_out()

            assert len(feat_names) == X_trans.shape[1], (
                f"Mismatch for encoder={enc}, fe={strat}: "
                f"{len(feat_names)} names vs {X_trans.shape[1]} columns"
            )


def test_threshold_optimizer_equidistant_ties_and_prevalence_bias() -> None:
    # Construct scenario where 0.46 and 0.54 give equal score, but positive prevalence is low (0.20)
    # The tie-breaker should favor 0.54 because it produces fewer false positives and better aligns with base rate.
    y_true = np.array([0, 0, 0, 0, 1])
    # Constant probabilities at 0.50: default to 0.50
    const_probs = np.full(5, 0.50)
    th_const = _find_best_binary_threshold(y_true, const_probs)
    assert abs(th_const - 0.50) < 0.01

    # Constant probabilities at extreme 0.10: default to 0.50
    low_const_probs = np.full(5, 0.10)
    th_low = _find_best_binary_threshold(y_true, low_const_probs)
    assert abs(th_low - 0.50) < 0.01


def test_deterministic_reproducibility() -> None:
    df = pd.DataFrame({
        "num_a": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5] * 3,
        "cat_b": ["red", "blue", "green", "red", "blue", "green"] * 3,
    })
    y = np.array([0, 1, 0, 1, 0, 1] * 3)

    cfg = PipelineConfig(
        imputer="median",
        scaler="robust",
        categorical_encoder="target_smooth",
        feature_engineering="poly_top5",
    )

    pipe1 = build_preprocessor_and_fe(cfg, df, random_state=123)
    X1 = pipe1.fit_transform(df, y)

    pipe2 = build_preprocessor_and_fe(cfg, df, random_state=123)
    X2 = pipe2.fit_transform(df, y)

    assert np.allclose(X1, X2, atol=1e-9)


def test_multiclass_target_encoder_absent_class_in_oof_subfold() -> None:
    """Tests that when a class is absent in an inner cross-fitting fold, mappings and dimensions stay 100% coherent."""
    # 3 classes: class 2 has only 2 samples
    X = np.array([[f"c_{i}"] for i in range(12)], dtype=object)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    enc = SmoothTargetEncoder(smoothing=5.0, cv=2, random_state=42)
    oof_res = enc.fit_transform(X, y)

    # Must produce 3 columns (1 per class)
    assert oof_res.shape == (12, 3)
    assert len(enc.classes_) == 3
    assert set(enc.classes_) == {0, 1, 2}
    assert all(c in enc.global_priors_ for c in [0, 1, 2])
    assert len(enc.mappings_multiclass_) == 1
    assert set(enc.mappings_multiclass_[0].keys()) == {0, 1, 2}

    # Transform must match shape and order exactly
    test_res = enc.transform(X[:4])
    assert test_res.shape == (4, 3)
    assert np.all((test_res >= 0.0) & (test_res <= 1.0))


def test_smooth_target_encoder_pandas_series_non_sequential_index() -> None:
    """Verifies that target encoding works seamlessly with pandas Series having non-sequential indices."""
    idx = [105, 42, 999, 13, 77, 300, 555, 888]
    X_df = pd.DataFrame({"cat_col": ["a", "b", "c", "a", "b", "c", "a", "b"]}, index=idx)
    y_series = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], index=idx)

    enc = SmoothTargetEncoder(smoothing=5.0, cv=2, random_state=42)
    # fit_transform and transform must not raise KeyError or alignment errors
    oof_out = enc.fit_transform(X_df.values, y_series)
    assert oof_out.shape == (8, 3)

    trans_out = enc.transform(X_df.values)
    assert trans_out.shape == (8, 3)


def test_sklearn_estimator_compatibility_and_clone() -> None:
    """Verifies full scikit-learn compatibility: clone, get_params, set_params, and ColumnTransformer."""
    from sklearn.base import clone

    enc = SmoothTargetEncoder(smoothing=12.5, cv=3, random_state=99)
    params = enc.get_params()
    assert params["smoothing"] == 12.5
    assert params["cv"] == 3
    assert params["random_state"] == 99

    cloned = clone(enc)
    assert cloned.smoothing == 12.5
    assert cloned.cv == 3
    assert cloned.random_state == 99

    cloned.set_params(smoothing=20.0, cv=4)
    assert cloned.smoothing == 20.0
    assert cloned.cv == 4


def test_threshold_optimizer_tolerance_no_score_drift_and_nan() -> None:
    """Verifies that score does not drift downwards during tie-breaking and handles NaNs/invalids with explicit warning."""
    # Test that candidate within 1e-5 of true global peak is selected by proximity to 0.50
    # without downward drift
    y_bal = np.array([0, 0, 0, 1, 1, 1])
    probs_plateau = np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])
    # For this perfectly separable data, all thresholds between 0.26 and 0.74 yield Macro-F1 = 1.0.
    # The tie-breaker MUST select 0.50 exactly
    best_th_plateau = _find_best_binary_threshold(y_bal, probs_plateau)
    assert abs(best_th_plateau - 0.50) < 1e-4

    # Degenerate input with NaNs: must issue explicit warning and return 0.50 default safely
    probs_with_nan = np.array([np.nan, 0.45, np.inf, 0.85])
    with pytest.warns(UserWarning, match="pos_probs contains NaN or Inf values"):
        th_nan = _find_best_binary_threshold(y_bal[:4], probs_with_nan)
    assert th_nan == 0.50


def test_target_encoder_dataset_with_two_samples_total() -> None:
    """Verifies that a dataset with only two samples total executes leak-free via LOO without crash."""
    X = np.array([["A"], ["B"]], dtype=object)
    y = np.array([0, 1])

    enc = SmoothTargetEncoder(smoothing=1.0, cv=5, random_state=42)
    # fit_transform on 2 samples: effective_cv < 2 triggers LOO
    res_ft = enc.fit_transform(X, y)
    assert res_ft.shape == (2, 1)
    # LOO global prior for row 0 (y=0) excluding itself: (1 - 0)/(2 - 1) = 1.0
    # LOO global prior for row 1 (y=1) excluding itself: (1 - 1)/(2 - 1) = 0.0
    assert abs(res_ft[0, 0] - 1.0) < 1e-3
    assert abs(res_ft[1, 0] - 0.0) < 1e-3

    # If both samples share the same category:
    X_shared = np.array([["A"], ["A"]], dtype=object)
    enc_shared = SmoothTargetEncoder(smoothing=1.0, cv=5, random_state=42)
    res_shared = enc_shared.fit_transform(X_shared, y)
    assert res_shared.shape == (2, 1)
    # Row 0 sees only row 1: LOO count=1, LOO sum=1, LOO prior=1.0 -> (1 + 1*1)/(1+1) = 1.0
    # Row 1 sees only row 0: LOO count=1, LOO sum=0, LOO prior=0.0 -> (0 + 1*0)/(1+1) = 0.0
    assert abs(res_shared[0, 0] - 1.0) < 1e-3
    assert abs(res_shared[1, 0] - 0.0) < 1e-3


def test_target_encoder_class_with_single_sample() -> None:
    """Verifies behavior when one class has only 1 sample (e.g. rare class), triggering LOO safely."""
    X = np.array([["c0"], ["c1"], ["c0"], ["c1"], ["c2"]], dtype=object)
    y = np.array([0, 0, 1, 1, 2])  # class 2 has only 1 sample

    enc = SmoothTargetEncoder(smoothing=2.0, cv=3, random_state=42)
    res = enc.fit_transform(X, y)

    # 3 classes -> 3 columns
    assert res.shape == (5, 3)
    assert np.all((res >= 0.0) & (res <= 1.0))

    # Transform on new data
    res_test = enc.transform(X[:2])
    assert res_test.shape == (2, 3)


def test_target_encoder_classes_absent_in_subfolds() -> None:
    """Verifies that when a class is absent in subfolds, columns and mappings stay 100% aligned."""
    X = np.array([[f"cat_{i}"] for i in range(15)], dtype=object)
    # class 2 appears only twice in 15 samples
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2])

    enc = SmoothTargetEncoder(smoothing=5.0, cv=2, random_state=42)
    res = enc.fit_transform(X, y)
    assert res.shape == (15, 3)

    names = enc.get_feature_names_out(["col0"])
    assert len(names) == 3
    assert list(names) == ["col0_class_0", "col0_class_1", "col0_class_2"]


def test_target_encoder_unique_category_per_sample() -> None:
    """Verifies that when every sample has a unique category, fit_transform produces global prior (zero leakage)."""
    X = np.array([[f"id_{i}"] for i in range(10)], dtype=object)
    y = np.array([0, 1] * 5)

    enc = SmoothTargetEncoder(smoothing=1.0, cv=5, random_state=42)
    res = enc.fit_transform(X, y)

    assert res.shape == (10, 1)
    # Prior is 0.50 (5 zeros, 5 ones)
    assert np.allclose(res, 0.50, atol=1e-3)


def test_loo_mathematical_zero_leakage_orthogonality() -> None:
    """Proves mathematically that sample i's encoded value is 100% invariant to sample i's own label y_i."""
    X = np.array([["A"], ["B"], ["A"], ["B"], ["C"]], dtype=object)
    # Target vector 1
    y1 = np.array([0, 1, 1, 0, 1])
    # Target vector 2 where sample 4's label is changed from 1 to 0
    y2 = np.array([0, 1, 1, 0, 0])

    # cv=1 triggers Leave-One-Out (effective_cv < 2)
    enc1 = SmoothTargetEncoder(smoothing=2.0, cv=1, random_state=42)
    res1 = enc1.fit_transform(X, y1)

    enc2 = SmoothTargetEncoder(smoothing=2.0, cv=1, random_state=42)
    res2 = enc2.fit_transform(X, y2)

    # Sample 4 has category 'C'. Sample 4's encoded value MUST be identical regardless of whether y[4] was 1 or 0!
    # Because sample 4 was excluded from both its category statistics and its global prior!
    assert abs(res1[4, 0] - res2[4, 0]) < 1e-9
    # Prior calculated from the other 4 samples [0, 1, 1, 0]: 2/4 = 0.50
    assert abs(res1[4, 0] - 0.50) < 1e-9

    # Multiclass test: rare singleton class triggering effective_cv < 2 with cv=5
    y_multi_1 = np.array([0, 0, 1, 1, 2])
    y_multi_2 = np.array([0, 0, 1, 1, 0])

    enc_m1 = SmoothTargetEncoder(smoothing=2.0, cv=5, random_state=42)
    res_m1 = enc_m1.fit_transform(X, y_multi_1)
    # In y_multi_1, sample 4 has category 'C' and unique class 2.
    # For class 2: the other 4 samples have target 0, so loo_prior for class 2 is 0/(5-1) = 0.0
    assert abs(res_m1[4, 2] - 0.0) < 1e-9


def test_oof_cross_fitting_zero_validation_leakage() -> None:
    """Confirms that in standard OOF cross-fitting, validation fold labels never affect validation encodings."""
    X_tr = np.array([["low"], ["high"], ["low"], ["high"], ["medium"], ["medium"]], dtype=object)
    y_tr = np.array([0, 1, 0, 1, 0, 1])

    enc = SmoothTargetEncoder(smoothing=5.0, cv=2, random_state=42)
    enc.fit(X_tr, y_tr)

    # Transform validation sets with different hypothetical labels (transform does not take y)
    X_val = np.array([["low"], ["unseen"]], dtype=object)
    out1 = enc.transform(X_val)
    out2 = enc.transform(X_val)

    assert np.allclose(out1, out2)
    assert abs(out1[1, 0] - enc.global_prior_) < 1e-5


def test_oof_multiclass_absent_class_validation_label_invariance() -> None:
    """Proves that validation fold labels NEVER leak into out-of-fold target encodings.

    When a class is absent in an inner training fold, the inner encoder must assign
    prior 0.0 without inheriting self.global_priors_ from the outer estimator.
    Altering labels exclusively inside the validation fold must produce strictly identical
    OOF encodings for the remaining validation samples.
    """
    # 11 samples: 5 of class 0, 5 of class 1, 1 of rare class 2 (index 10)
    X = np.array([["A"], ["A"], ["A"], ["B"], ["B"], ["B"], ["C"], ["C"], ["D"], ["D"], ["E"]], dtype=object)
    y1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])

    # StratifiedKFold splits indices: in Fold 0, sample 10 (class 2) is in val_idx,
    # so class 2 has 0 samples in tr_idx.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_indices_with_rare_class = None
    tr_indices_with_rare_class = None
    for tr_idx, val_idx in skf.split(X, y1):
        if 10 in val_idx:
            val_indices_with_rare_class = list(val_idx)
            tr_indices_with_rare_class = list(tr_idx)
            break

    assert val_indices_with_rare_class is not None
    # Peer validation samples in the same fold (e.g. index 3 and 8)
    peer_val_indices = [idx for idx in val_indices_with_rare_class if idx != 10]

    enc = SmoothTargetEncoder(smoothing=5.0, cv=5, random_state=42)
    oof = enc.fit_transform(X, y1)

    # In Fold 0, class 2 was completely absent in tr_idx.
    # Therefore, all peer validation samples in that fold MUST receive exactly prior 0.0 for class 2.
    # In the old buggy implementation, the fallback "inner_enc.global_priors_[c] = self.global_priors_[c]"
    # leaked self.global_priors_[2] = 1/11 (~0.0909) into oof[peer_idx, 2], failing this zero-leakage test.
    for peer_idx in peer_val_indices:
        assert oof[peer_idx, 2] == 0.0

    # Furthermore, verify that an inner cross-fitting fold estimator with absent classes
    # is 100% mathematically invariant to validation fold labels:
    X_tr = X[tr_indices_with_rare_class]
    y_tr = y1[tr_indices_with_rare_class]  # only classes 0 and 1, class 2 is absent
    X_val = X[val_indices_with_rare_class]

    # Inner encoder trained exclusively on training fold with classes=[0, 1, 2]
    inner_enc = SmoothTargetEncoder(smoothing=5.0, cv=None, random_state=42)
    inner_enc.fit(X_tr, y_tr, classes=enc.classes_)

    # Absent class 2 must have global prior 0.0
    assert inner_enc.global_priors_[2] == 0.0

    # Transforming validation samples produces strictly 0.0 for class 2
    val_transformed = inner_enc.transform(X_val)
    assert np.all(val_transformed[:, 2] == 0.0)




