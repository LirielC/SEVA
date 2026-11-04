"""5-Layer Declarative Pipeline Builder with strict fold isolation (zero data leakage)."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from autoresearcher.schemas import PipelineConfig

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


class SmoothTargetEncoder(BaseEstimator, TransformerMixin):
    """Leak-free smoothed target encoding for categorical features (supports binary & multiclass with OOF cross-fitting)."""

    def __init__(self, smoothing: float = 10.0, cv: Optional[int] = 5, random_state: int = 42) -> None:
        self.smoothing = smoothing
        self.cv = cv
        self.random_state = random_state
        self.global_prior_: float = 0.0
        self.mappings_: List[dict] = []
        self.mappings_multiclass_: List[dict] = []
        self.global_priors_: Dict[object, float] = {}
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
    ) -> "SmoothTargetEncoder":
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.n_features_in_ = X_arr.shape[1]
        if y is None:
            self.classes_ = None
            self.global_prior_ = 0.5
            self.mappings_ = [{} for _ in range(self.n_features_in_)]
            self.mappings_multiclass_ = []
            self.global_priors_ = {}
            return self

        y_arr = np.asarray(y).reshape(-1)
        if classes is not None:
            self.classes_ = np.asarray(classes)
        else:
            self.classes_ = np.unique(y_arr)

        if len(self.classes_) > 2:
            # Multiclass decomposition: K smoothed binary target probabilities
            self.global_priors_ = {
                c: float(np.mean(y_arr == c)) if len(y_arr) > 0 and np.any(y_arr == c) else 0.0
                for c in self.classes_
            }
            self.mappings_multiclass_ = []
            for col_idx in range(self.n_features_in_):
                col_vals = X_arr[:, col_idx]
                col_dict = {}
                for c in self.classes_:
                    y_bin = (y_arr == c).astype(float)
                    prior_c = self.global_priors_[c]
                    if np.sum(y_bin) > 0:
                        df_tmp = pd.DataFrame({"val": col_vals, "y": y_bin})
                        stats = df_tmp.groupby("val")["y"].agg(["count", "mean"])
                        counts = stats["count"]
                        means = stats["mean"]
                        smooth = (counts * means + self.smoothing * prior_c) / (counts + self.smoothing)
                        col_dict[c] = smooth.to_dict()
                    else:
                        col_dict[c] = {}
                self.mappings_multiclass_.append(col_dict)
            self.mappings_ = []
            self.global_prior_ = 0.0
        else:
            # Binary classification: probability of positive class
            pos_label = self.classes_[-1] if len(self.classes_) > 0 else 1
            y_bin = (y_arr == pos_label).astype(float)
            self.global_prior_ = float(np.mean(y_bin)) if len(y_bin) > 0 else 0.5
            self.mappings_ = []
            self.mappings_multiclass_ = []
            self.global_priors_ = {}
            for col_idx in range(self.n_features_in_):
                col_vals = X_arr[:, col_idx]
                df_tmp = pd.DataFrame({"val": col_vals, "y": y_bin})
                stats = df_tmp.groupby("val")["y"].agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                smooth = (counts * means + self.smoothing * self.global_prior_) / (counts + self.smoothing)
                self.mappings_.append(smooth.to_dict())
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        **fit_params,
    ) -> np.ndarray:
        """Fit target encoder and return out-of-fold cross-fitted transformations to eliminate in-sample target leakage."""
        if y is None:
            self.fit(X, None)
            return self.transform(X)

        y_arr = np.asarray(y).reshape(-1)
        classes_arg = classes if classes is not None else fit_params.get("classes", None)
        self.fit(X, y_arr, classes=classes_arg)

        if self.cv is None:
            return self.transform(X)

        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = len(X_arr)
        classes_unique, class_counts = np.unique(y_arr, return_counts=True)
        max_class_count = int(np.max(class_counts)) if len(class_counts) > 0 else 0
        effective_cv = min(self.cv, max_class_count, n_samples)
        if effective_cv < 2:
            # When effective_cv < 2 (e.g., singleton class or tiny dataset), StratifiedKFold cannot form >=2 folds.
            # Strictly prevent in-sample target leakage by applying analytical Leave-One-Out (LOO) smoothing:
            # for each sample i, sample i's own target is subtracted from the category statistics.
            if n_samples <= 1:
                # 0 or 1 sample: no out-of-sample data exists, return neutral global prior
                if self.classes_ is not None and len(self.classes_) > 2:
                    out = np.zeros((n_samples, self.n_features_in_ * len(self.classes_)), dtype=float)
                    for col_idx in range(self.n_features_in_):
                        for c_idx, c in enumerate(self.classes_):
                            out[:, col_idx * len(self.classes_) + c_idx] = self.global_priors_[c]
                    return out
                return np.full((n_samples, self.n_features_in_), self.global_prior_, dtype=float)

            # Analytical Leave-One-Out: subtract row i's target from category sum and count,
            # AND compute leave-one-out global priors excluding sample i from the total dataset prior
            if self.classes_ is not None and len(self.classes_) > 2:
                out_cols = []
                for col_idx in range(self.n_features_in_):
                    col_vals = X_arr[:, col_idx]
                    for c in self.classes_:
                        y_bin = (y_arr == c).astype(float)
                        total_c = float(np.sum(y_bin))
                        loo_prior_c = (total_c - y_bin) / float(n_samples - 1)

                        df_tmp = pd.DataFrame({"val": col_vals, "y": y_bin})
                        cat_counts = df_tmp.groupby("val")["y"].transform("count").values
                        cat_sums = df_tmp.groupby("val")["y"].transform("sum").values
                        loo_counts = cat_counts - 1.0
                        loo_sums = cat_sums - y_bin
                        loo_encoded = np.where(
                            loo_counts > 0,
                            (loo_sums + self.smoothing * loo_prior_c) / (loo_counts + self.smoothing),
                            loo_prior_c,
                        )
                        out_cols.append(loo_encoded.reshape(-1, 1))
                return np.hstack(out_cols)
            else:
                pos_label = self.classes_[-1] if len(self.classes_) > 0 else 1
                y_bin = (y_arr == pos_label).astype(float)
                total_pos = float(np.sum(y_bin))
                loo_global_prior = (total_pos - y_bin) / float(n_samples - 1)

                out = np.zeros((n_samples, self.n_features_in_), dtype=float)
                for col_idx in range(self.n_features_in_):
                    col_vals = X_arr[:, col_idx]
                    df_tmp = pd.DataFrame({"val": col_vals, "y": y_bin})
                    cat_counts = df_tmp.groupby("val")["y"].transform("count").values
                    cat_sums = df_tmp.groupby("val")["y"].transform("sum").values
                    loo_counts = cat_counts - 1.0
                    loo_sums = cat_sums - y_bin
                    loo_encoded = np.where(
                        loo_counts > 0,
                        (loo_sums + self.smoothing * loo_global_prior) / (loo_counts + self.smoothing),
                        loo_global_prior,
                    )
                    out[:, col_idx] = loo_encoded
                return out

        skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)

        # Allocate OOF array
        n_out_cols = (
            self.n_features_in_ * len(self.classes_)
            if (self.classes_ is not None and len(self.classes_) > 2)
            else self.n_features_in_
        )
        oof_encoded = np.zeros((n_samples, n_out_cols), dtype=float)

        for tr_idx, val_idx in skf.split(X_arr, y_arr):
            inner_enc = SmoothTargetEncoder(smoothing=self.smoothing, cv=None, random_state=self.random_state)
            inner_enc.fit(X_arr[tr_idx], y_arr[tr_idx], classes=self.classes_)
            oof_encoded[val_idx] = inner_enc.transform(X_arr[val_idx])

        return oof_encoded

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.classes_ is not None and len(self.classes_) > 2:
            out_cols = []
            for col_idx in range(self.n_features_in_):
                col_vals = X_arr[:, col_idx]
                for c in self.classes_:
                    mapping = self.mappings_multiclass_[col_idx][c]
                    prior = self.global_priors_[c]
                    col_encoded = [float(mapping.get(v, prior)) for v in col_vals]
                    out_cols.append(np.array(col_encoded, dtype=float).reshape(-1, 1))
            return np.hstack(out_cols) if out_cols else np.empty((X_arr.shape[0], 0), dtype=float)

        out = np.full((X_arr.shape[0], self.n_features_in_), self.global_prior_, dtype=float)
        for col_idx in range(self.n_features_in_):
            mapping = self.mappings_[col_idx]
            col_vals = X_arr[:, col_idx]
            out[:, col_idx] = [float(mapping.get(v, self.global_prior_)) for v in col_vals]
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names matching binary or multiclass column expansion."""
        if input_features is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            names = list(input_features)

        if self.classes_ is not None and len(self.classes_) > 2:
            out_names = [f"{col}_class_{c}" for col in names for c in self.classes_]
            return np.array(out_names, dtype=object)
        return np.array(names, dtype=object)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Layer 3: Supervised feature interactions or selection fit strictly inside each CV fold."""

    def __init__(self, strategy: str = "none", random_state: int = 42) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.top_indices_: np.ndarray = np.array([], dtype=int)
        self.selector_: Optional[SelectPercentile] = None
        self.keep_mask_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureEngineeringTransformer":
        X_num = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        self.n_features_in_ = X_num.shape[1]
        if self.n_features_in_ == 0 or self.strategy == "none":
            return self

        if self.strategy in ("poly_top5", "ratios_top5"):
            k = min(5, self.n_features_in_)
            if y is not None and len(np.unique(y)) > 1:
                f_vals, _ = f_classif(X_num, y)
                f_vals = np.nan_to_num(f_vals, nan=0.0)
                self.top_indices_ = np.argsort(f_vals)[::-1][:k]
            else:
                variances = np.var(X_num, axis=0)
                self.top_indices_ = np.argsort(variances)[::-1][:k]

        elif self.strategy == "mutual_info_k80" and y is not None:
            self.selector_ = SelectPercentile(score_func=f_classif, percentile=80)
            self.selector_.fit(X_num, y)

        elif self.strategy == "tree_importance_k75" and y is not None:
            et = ExtraTreesClassifier(n_estimators=35, max_depth=6, random_state=self.random_state, n_jobs=1)
            et.fit(X_num, y)
            importances = et.feature_importances_
            k_keep = max(2, int(np.ceil(0.75 * self.n_features_in_)))
            top_k_idx = np.argsort(importances)[::-1][:k_keep]
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[top_k_idx] = True
            self.keep_mask_ = mask

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_num = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if X_num.shape[1] == 0 or self.strategy == "none":
            return X_num

        if self.strategy == "poly_top5" and len(self.top_indices_) > 0:
            sub = X_num[:, self.top_indices_]
            extras = [sub ** 2]
            for i in range(sub.shape[1]):
                for j in range(i + 1, sub.shape[1]):
                    extras.append((sub[:, i] * sub[:, j]).reshape(-1, 1))
            return np.hstack([X_num] + extras)

        if self.strategy == "ratios_top5" and len(self.top_indices_) > 0:
            sub = X_num[:, self.top_indices_]
            extras = []
            for i in range(sub.shape[1]):
                for j in range(i + 1, sub.shape[1]):
                    ratio = (sub[:, i] / (np.abs(sub[:, j]) + 1.0)).reshape(-1, 1)
                    diff = (sub[:, i] - sub[:, j]).reshape(-1, 1)
                    extras.extend([ratio, diff])
            return np.hstack([X_num] + extras) if extras else X_num

        if self.strategy == "mutual_info_k80" and self.selector_ is not None:
            transformed = self.selector_.transform(X_num)
            return transformed if transformed.shape[1] > 0 else X_num

        if self.strategy == "tree_importance_k75" and self.keep_mask_ is not None:
            return X_num[:, self.keep_mask_]

        return X_num

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names with engineered interaction and pruned feature names."""
        if input_features is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            names = list(input_features)

        if self.n_features_in_ == 0 or self.strategy == "none":
            return np.array(names, dtype=object)

        if self.strategy == "poly_top5" and len(self.top_indices_) > 0:
            extras = [f"{names[i]}_sq" for i in self.top_indices_]
            for i in range(len(self.top_indices_)):
                for j in range(i + 1, len(self.top_indices_)):
                    idx_i, idx_j = self.top_indices_[i], self.top_indices_[j]
                    extras.append(f"{names[idx_i]}x{names[idx_j]}")
            return np.array(names + extras, dtype=object)

        if self.strategy == "ratios_top5" and len(self.top_indices_) > 0:
            extras = []
            for i in range(len(self.top_indices_)):
                for j in range(i + 1, len(self.top_indices_)):
                    idx_i, idx_j = self.top_indices_[i], self.top_indices_[j]
                    extras.append(f"ratio_{names[idx_i]}_{names[idx_j]}")
                    extras.append(f"diff_{names[idx_i]}_{names[idx_j]}")
            return np.array(names + extras, dtype=object)

        if self.strategy == "mutual_info_k80" and self.selector_ is not None:
            support = self.selector_.get_support()
            if np.any(support):
                return np.array([names[i] for i in range(len(names)) if support[i]], dtype=object)
            return np.array(names, dtype=object)

        if self.strategy == "tree_importance_k75" and self.keep_mask_ is not None:
            if np.any(self.keep_mask_):
                return np.array([names[i] for i in range(len(names)) if self.keep_mask_[i]], dtype=object)
            return np.array(names, dtype=object)

        return np.array(names, dtype=object)


def build_preprocessor_and_fe(
    config: PipelineConfig,
    X_sample: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Build Layers 1, 2, and 3 as a scikit-learn Pipeline."""
    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_sample.columns if c not in num_cols]

    # Layer 1: Numeric Imputation & Scaling
    num_steps: list[Tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy=config.imputer)),
    ]
    if config.scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif config.scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif config.scaler == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))

    num_pipeline = Pipeline(num_steps)

    # Layer 2: Categorical Imputation & Encoding
    cat_steps: list[Tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
    if config.categorical_encoder == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif config.categorical_encoder == "ordinal":
        cat_steps.append(
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        )
    elif config.categorical_encoder == "target_smooth":
        cat_steps.append(("encoder", SmoothTargetEncoder(smoothing=10.0, cv=5, random_state=random_state)))

    cat_pipeline = Pipeline(cat_steps)

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    # Layer 3: Feature Engineering & Selection
    fe_step = FeatureEngineeringTransformer(
        strategy=config.feature_engineering,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocess", col_transformer),
        ("feature_engineering", fe_step),
    ])


def apply_fold_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Layer 4: Apply class resampling strictly inside the training fold (never on validation/test)."""
    if strategy not in ("random_oversample", "random_undersample"):
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) <= 1:
        return X_train, y_train

    target_count = int(counts.max()) if strategy == "random_oversample" else int(counts.min())
    indices_list = []
    for cls in classes:
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) == target_count:
            indices_list.append(cls_idx)
        elif len(cls_idx) < target_count:
            extra = rng.choice(cls_idx, size=target_count - len(cls_idx), replace=True)
            indices_list.append(np.concatenate([cls_idx, extra]))
        else:
            sub = rng.choice(cls_idx, size=target_count, replace=False)
            indices_list.append(sub)

    all_idx = np.concatenate(indices_list)
    rng.shuffle(all_idx)
    return X_train[all_idx], y_train[all_idx]


def instantiate_classifier(
    config: PipelineConfig,
    n_classes: int,
    random_state: int = 42,
) -> BaseEstimator:
    """Layer 5: Instantiate model family with normalized hyperparameters and regularization."""
    use_balanced_weights = config.imbalance_strategy == "class_weight_balanced"

    if config.model_family == "random_forest":
        max_samples = None if config.subsample >= 0.99 else float(config.subsample)
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced_subsample" if use_balanced_weights else None,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=1,
        )

    if config.model_family == "xgboost":
        if HAS_XGBOOST:
            objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
            return XGBClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                reg_lambda=config.regularization,
                subsample=config.subsample,
                min_child_weight=float(config.min_samples_leaf),
                objective=objective,
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
        return HistGradientBoostingClassifier(
            max_iter=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            l2_regularization=config.regularization,
            min_samples_leaf=max(5, config.min_samples_leaf * 3),
            class_weight="balanced" if use_balanced_weights else None,
            random_state=random_state,
        )

    # CatBoost family
    if HAS_CATBOOST:
        loss_fn = "Logloss" if n_classes == 2 else "MultiClass"
        cb_kwargs = {
            "iterations": config.n_estimators,
            "depth": min(8, config.max_depth),
            "learning_rate": config.learning_rate,
            "l2_leaf_reg": config.regularization,
            "loss_function": loss_fn,
            "random_seed": random_state,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
        if use_balanced_weights:
            cb_kwargs["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(**cb_kwargs)

    return GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=min(8, config.max_depth),
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        min_samples_leaf=config.min_samples_leaf,
        random_state=random_state,
    )


def fit_estimator_with_optional_weights(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> BaseEstimator:
    """Fit classifier, passing sample_weight when model family needs explicit balanced weights."""
    if config.imbalance_strategy == "class_weight_balanced" and config.model_family == "xgboost":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        try:
            estimator.fit(X_train, y_train, sample_weight=sample_weights)
            return estimator
        except TypeError:
            pass
    estimator.fit(X_train, y_train)
    return estimator
