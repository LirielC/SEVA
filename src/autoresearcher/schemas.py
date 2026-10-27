"""Pydantic schemas for declarative ML pipelines, diagnostics, and agent decisions."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ImputerType = Literal["median", "mean"]
ScalerType = Literal["none", "standard", "robust", "minmax"]
CategoricalEncoderType = Literal["onehot", "ordinal", "target_smooth"]
FeatureEngineeringType = Literal[
    "none",
    "poly_top5",
    "ratios_top5",
    "mutual_info_k80",
    "tree_importance_k75",
]
ImbalanceStrategyType = Literal[
    "none",
    "class_weight_balanced",
    "random_oversample",
    "random_undersample",
]
ModelFamilyType = Literal["random_forest", "xgboost", "catboost"]
ExplorationModeType = Literal["baseline", "exploit_refinement", "explore_new_family"]


class PipelineConfig(BaseModel):
    """5-Layer Declarative CASH + Pipeline Configuration shared across all methods."""

    # Layer 1: Numeric Imputation & Scaling
    imputer: ImputerType = Field(default="median", description="Numeric missing value strategy")
    scaler: ScalerType = Field(default="none", description="Numeric feature scaling strategy")

    # Layer 2: Categorical Encoding
    categorical_encoder: CategoricalEncoderType = Field(
        default="onehot",
        description="Categorical feature encoding strategy",
    )

    # Layer 3: Feature Engineering & Selection
    feature_engineering: FeatureEngineeringType = Field(
        default="none",
        description="Feature interaction creation or supervised selection inside CV fold",
    )

    # Layer 4: Class Imbalance & Threshold Post-Processing
    imbalance_strategy: ImbalanceStrategyType = Field(
        default="none",
        description="Class balancing strategy applied strictly to training folds",
    )
    threshold_tuning: bool = Field(
        default=False,
        description="Optimize decision threshold on validation probabilities (binary tasks)",
    )

    # Layer 5: Model Family & Regularized Hyperparameters
    model_family: ModelFamilyType = Field(
        default="random_forest",
        description="Classifier model family",
    )
    n_estimators: int = Field(default=150, ge=50, le=500, description="Number of trees/iterations")
    max_depth: int = Field(default=6, ge=3, le=12, description="Maximum tree depth")
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.3,
        description="Boosting learning rate (used by xgboost and catboost)",
    )
    regularization: float = Field(
        default=1.0,
        ge=0.01,
        le=20.0,
        description="L2 regularization strength (reg_lambda / l2_leaf_reg)",
    )
    subsample: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Row subsampling ratio per tree",
    )
    min_samples_leaf: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum samples per leaf (controls overfitting in trees)",
    )

    def canonical_dict(self) -> Dict[str, object]:
        """Return a normalized dictionary representation so irrelevant params don't split hashes."""
        data = self.model_dump()
        if self.model_family == "random_forest":
            # Random forest does not use boosting learning_rate
            data["learning_rate"] = 0.1
        return data

    def config_hash(self) -> str:
        """Compute deterministic SHA-256 short hash for anti-duplication guardrails."""
        payload = json.dumps(self.canonical_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def diff_from(self, other: "PipelineConfig") -> List[str]:
        """List human-readable changes between another configuration and this one."""
        d1 = other.canonical_dict()
        d2 = self.canonical_dict()
        diffs: List[str] = []
        for key, val2 in d2.items():
            val1 = d1.get(key)
            if val1 != val2:
                diffs.append(f"{key}: {val1} -> {val2}")
        return diffs

    @classmethod
    def default_baseline(cls) -> "PipelineConfig":
        """Standard unregularized baseline used as Trial 0 across runs."""
        return cls(
            imputer="median",
            scaler="none",
            categorical_encoder="onehot",
            feature_engineering="none",
            imbalance_strategy="none",
            threshold_tuning=False,
            model_family="random_forest",
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            regularization=1.0,
            subsample=1.0,
            min_samples_leaf=1,
        )


class DatasetProfile(BaseModel):
    """Metadata and statistical profile of a tabular classification dataset."""

    name: str
    openml_id: Optional[int] = None
    n_samples_total: int
    n_samples_dev: int
    n_samples_test: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_classes: int
    class_distribution: Dict[str, float]
    imbalance_ratio: float = Field(description="Ratio of largest class to smallest class")
    missing_ratio: float
    high_cardinality_cols: List[str] = Field(default_factory=list)


class ConfusedPair(BaseModel):
    """Represents a frequent confusion between a true class and predicted class."""

    true_class: str
    predicted_class: str
    error_count: int
    error_rate_within_true_class: float


class DiagnosticReport(BaseModel):
    """Rich error analysis computed exclusively on Cross-Validation folds (zero test leakage)."""

    val_macro_f1: float
    val_std_f1: float
    train_macro_f1: float
    overfit_gap: float = Field(description="train_macro_f1 - val_macro_f1")
    val_balanced_accuracy: float
    per_class_f1: Dict[str, float]
    worst_class_label: str
    worst_class_f1: float
    best_class_f1: float
    top_confused_pairs: List[ConfusedPair] = Field(default_factory=list)
    top_features: List[str] = Field(default_factory=list)
    primary_pathology: Literal[
        "severe_overfitting",
        "minority_class_collapse",
        "underfitting_low_capacity",
        "feature_noise_or_representation",
        "well_balanced_needs_fine_tuning",
    ]
    diagnostic_summary_text: str


class AgentDecision(BaseModel):
    """Structured scientific reasoning emitted by the AutoResearcher Agent before each trial."""

    bottleneck_diagnosis: str = Field(
        description="Diagnosis of main error pattern from CV metrics, train-val gap, and confusion matrix"
    )
    scientific_hypothesis: str = Field(
        description="Falsifiable hypothesis: If we change X, metric Y will improve because Z"
    )
    exploration_mode: ExplorationModeType = Field(
        default="exploit_refinement",
        description="Whether refining the best config or exploring a new model family",
    )
    changed_components: List[str] = Field(
        default_factory=list,
        description="Explicit list of pipeline dimensions modified",
    )
    config: PipelineConfig


class TrialRecord(BaseModel):
    """Complete audit log of a single experiment trial."""

    trial_index: int
    strategy_name: str
    dataset_name: str
    seed: int
    config: PipelineConfig
    config_hash: str
    diagnostics: DiagnosticReport
    best_val_macro_f1_so_far: float
    elapsed_seconds: float
    agent_decision: Optional[AgentDecision] = None
    was_new_best: bool = False


class MethodBenchmarkResult(BaseModel):
    """Result of running one optimization strategy on one dataset with one seed."""

    strategy_name: str
    dataset_name: str
    seed: int
    budget: int
    best_val_macro_f1: float
    holdout_test_macro_f1: float
    baseline_val_macro_f1: float
    improvement_over_baseline: float
    trials_to_95pct_peak: int
    total_elapsed_seconds: float
    best_config: PipelineConfig
    trials: List[TrialRecord]
