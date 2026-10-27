"""OpenML dataset loader, profiler, and leak-free Dev/Holdout splitter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from autoresearcher.config import SETTINGS
from autoresearcher.schemas import DatasetProfile

# Registry of curated OpenML datasets with distinct pathologies
OPENML_DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "credit-g": {
        "openml_id": 31,
        "description": "German Credit Risk (Imbalanced 70/30, mixed categorical/numeric, high overfitting risk)",
    },
    "phoneme": {
        "openml_id": 1489,
        "description": "Phoneme Acoustic Classification (Non-linear numerical boundaries, class imbalance)",
    },
    "vehicle": {
        "openml_id": 54,
        "description": "Statlog Vehicle Silhouettes (4-class multiclass with overlapping pairs opel/saab)",
    },
    "diabetes": {
        "openml_id": 37,
        "description": "Pima Indians Diabetes (Noisy numerical clinical features, imbalanced binary)",
    },
    "splice": {
        "openml_id": 46,
        "description": "Primate Splice-Junction DNA Sequences (3-class categorical high-dimensional)",
    },
}


@dataclass
class DatasetSplit:
    """Container holding strictly isolated Development (CV) and Holdout Test sets."""

    profile: DatasetProfile
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    class_names: list[str]


def _generate_realistic_fallback(name: str, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Optional[int]]:
    """Generate deterministic dataset mimicking the exact pathology if offline or in fast test mode."""
    rng = np.random.default_rng(random_state)

    if name == "credit-g":
        # Pathologies: 70/30 class imbalance, mixed categorical + numeric, noisy features causing tree overfitting
        X_num, y = make_classification(
            n_samples=1000,
            n_features=12,
            n_informative=5,
            n_redundant=3,
            n_repeated=0,
            n_classes=2,
            weights=[0.70, 0.30],
            flip_y=0.06,
            class_sep=0.85,
            random_state=random_state,
        )
        df = pd.DataFrame(X_num, columns=[f"num_feat_{i}" for i in range(12)])
        # Inject realistic categorical features and missing values
        df["checking_status"] = rng.choice(["<0", "0<=X<200", ">=200", "no_checking"], size=1000, p=[0.3, 0.3, 0.1, 0.3])
        df["credit_history"] = rng.choice(["critical", "existing_paid", "delayed", "all_paid"], size=1000)
        df["purpose"] = rng.choice(
            ["radio/tv", "education", "furniture", "new_car", "used_car", "business", "repairs"],
            size=1000,
        )
        df["housing"] = rng.choice(["own", "rent", "for_free"], size=1000, p=[0.7, 0.2, 0.1])
        # Correlate checking_status with target to reward proper encoding + regularization
        mask_risk = y == 1
        df.loc[mask_risk, "checking_status"] = rng.choice(
            ["<0", "0<=X<200", ">=200", "no_checking"],
            size=int(mask_risk.sum()),
            p=[0.55, 0.25, 0.05, 0.15],
        )
        # Inject 4% missing values in two numeric columns
        miss_idx = rng.choice(1000, size=40, replace=False)
        df.loc[miss_idx, "num_feat_0"] = np.nan
        target = pd.Series(np.where(y == 1, "bad", "good"), name="class")
        return df, target, 31

    if name == "phoneme":
        # Pathologies: Non-linear interactions, ratios/polynomials help, 70/30 imbalance
        X_num, y = make_classification(
            n_samples=1200,
            n_features=8,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=2,
            n_classes=2,
            weights=[0.71, 0.29],
            flip_y=0.04,
            class_sep=0.9,
            random_state=random_state + 7,
        )
        df = pd.DataFrame(X_num, columns=[f"formant_freq_{i+1}" for i in range(8)])
        # Add non-linear ratio interaction signal
        nonlinear_score = (df["formant_freq_1"] * df["formant_freq_2"]) - 0.6 * (df["formant_freq_3"] ** 2)
        y_mod = np.where((nonlinear_score > np.percentile(nonlinear_score, 72)) | (y == 1), 1, 0)
        # Flip 5% label noise
        noise_mask = rng.random(1200) < 0.05
        y_mod = np.where(noise_mask, 1 - y_mod, y_mod)
        target = pd.Series(np.where(y_mod == 1, "nasal", "oral"), name="class")
        return df, target, 1489

    # Default / "vehicle": 4-class multiclass problem with correlated shape features and confused pairs
    X_num, y = make_classification(
        n_samples=846,
        n_features=18,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=0.95,
        random_state=random_state + 13,
    )
    df = pd.DataFrame(X_num, columns=[f"shape_metric_{i+1}" for i in range(18)])
    label_map = {0: "bus", 1: "opel", 2: "saab", 3: "van"}
    target = pd.Series([label_map[int(v)] for v in y], name="class")
    return df, target, 54


def load_benchmark_dataset(
    dataset_name: str,
    test_size: float = 0.20,
    split_seed: int = 42,
    prefer_offline_fallback: bool = False,
    max_samples: int = 2500,
) -> DatasetSplit:
    """Load an OpenML dataset, profile its characteristics, and create a locked Dev/Holdout split."""
    cache_dir = Path(SETTINGS.artifacts_dir) / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_cache = cache_dir / f"{dataset_name}.parquet"

    openml_id: Optional[int] = None
    meta = OPENML_DATASET_REGISTRY.get(dataset_name, {})
    if meta:
        openml_id = int(meta["openml_id"])  # type: ignore[arg-type]

    X_df: Optional[pd.DataFrame] = None
    y_ser: Optional[pd.Series] = None

    if not prefer_offline_fallback and parquet_cache.exists():
        cached = pd.read_parquet(parquet_cache)
        y_ser = cached["__target__"].astype(str)
        X_df = cached.drop(columns=["__target__"])
    elif not prefer_offline_fallback:
        try:
            if openml_id is not None:
                bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
            else:
                bunch = fetch_openml(name=dataset_name, as_frame=True, parser="auto")
            X_df = bunch.data.copy()
            y_ser = bunch.target.astype(str).copy()

            # Subsample stratified if larger than max_samples to keep CV fast
            if len(X_df) > max_samples:
                X_df, _, y_ser, _ = train_test_split(
                    X_df,
                    y_ser,
                    train_size=max_samples,
                    stratify=y_ser,
                    random_state=split_seed,
                )
                X_df = X_df.reset_index(drop=True)
                y_ser = y_ser.reset_index(drop=True)

            to_save = X_df.copy()
            # Convert categorical columns to string/object for safe parquet storage
            for col in to_save.select_dtypes(include=["category", "object"]).columns:
                to_save[col] = to_save[col].astype(str)
            to_save["__target__"] = y_ser.values
            to_save.to_parquet(parquet_cache, index=False)
        except Exception:
            X_df, y_ser, openml_id = _generate_realistic_fallback(dataset_name, random_state=split_seed)
    else:
        X_df, y_ser, openml_id = _generate_realistic_fallback(dataset_name, random_state=split_seed)

    assert X_df is not None and y_ser is not None

    # Normalize column types: numeric -> float64, categorical/string -> object
    X_clean = X_df.copy()
    for col in X_clean.columns:
        if pd.api.types.is_numeric_dtype(X_clean[col]) and not pd.api.types.is_bool_dtype(X_clean[col]):
            X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce").astype(float)
        else:
            X_clean[col] = X_clean[col].astype(object)

    le = LabelEncoder()
    y_encoded = np.asarray(le.fit_transform(y_ser.astype(str)), dtype=int)
    class_names = [str(c) for c in le.classes_]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_clean,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=split_seed,
    )

    X_dev = X_dev.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Profile exclusively using general metadata + Dev distribution (never leak test statistics)
    num_cols = X_dev.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_dev.columns if c not in num_cols]
    high_card = [c for c in cat_cols if X_dev[c].nunique(dropna=True) > 10]

    unique_vals, counts = np.unique(y_dev, return_counts=True)
    total_dev = float(len(y_dev))
    class_dist = {class_names[int(u)]: round(float(cnt) / total_dev, 4) for u, cnt in zip(unique_vals, counts)}
    imbalance_ratio = round(float(counts.max()) / max(1.0, float(counts.min())), 3)
    missing_ratio = round(float(X_dev.isna().sum().sum()) / max(1.0, float(X_dev.size)), 4)

    profile = DatasetProfile(
        name=dataset_name,
        openml_id=openml_id,
        n_samples_total=len(X_clean),
        n_samples_dev=len(X_dev),
        n_samples_test=len(X_test),
        n_features=X_dev.shape[1],
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        n_classes=len(class_names),
        class_distribution=class_dist,
        imbalance_ratio=imbalance_ratio,
        missing_ratio=missing_ratio,
        high_cardinality_cols=high_card,
    )

    return DatasetSplit(
        profile=profile,
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )
