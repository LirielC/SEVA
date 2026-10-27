"""Optuna TPE (Tree-structured Parzen Estimator) Bayesian Optimization baseline."""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np
import optuna

from autoresearcher.baselines.space_catalog import SEARCH_SPACE_CATALOG, sample_random_config
from autoresearcher.schemas import PipelineConfig, TrialRecord

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPEStrategy:
    """Uses Optuna's TPESampler over the exact same 5-layer CASH space, seeing only scalar val_macro_f1."""

    name: str = "Optuna (TPE)"

    def __init__(self, seed: int = 42, n_classes: int = 2) -> None:
        self.seed = seed
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.seen_hashes: Set[str] = set()
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=6, multivariate=True)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending_trial: Optional[optuna.trial.Trial] = None
        self._last_reported_idx: int = -1

    def _sync_history(self, history: List[TrialRecord]) -> None:
        """Report the scalar score of the latest trial back to Optuna's study."""
        if not history:
            return
        latest = history[-1]
        if latest.trial_index <= self._last_reported_idx:
            return

        if self._pending_trial is not None:
            self.study.tell(self._pending_trial, latest.diagnostics.val_macro_f1)
            self._pending_trial = None
        else:
            # Enqueue Trial 0 baseline result into study
            distributions: Dict[str, optuna.distributions.BaseDistribution] = {
                "imputer": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["imputer"]),
                "scaler": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["scaler"]),
                "categorical_encoder": optuna.distributions.CategoricalDistribution(
                    SEARCH_SPACE_CATALOG["categorical_encoder"]
                ),
                "feature_engineering": optuna.distributions.CategoricalDistribution(
                    SEARCH_SPACE_CATALOG["feature_engineering"]
                ),
                "imbalance_strategy": optuna.distributions.CategoricalDistribution(
                    SEARCH_SPACE_CATALOG["imbalance_strategy"]
                ),
                "threshold_tuning": optuna.distributions.CategoricalDistribution(
                    SEARCH_SPACE_CATALOG["threshold_tuning"] if self.n_classes == 2 else [False]
                ),
                "model_family": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["model_family"]),
                "n_estimators": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["n_estimators"]),
                "max_depth": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["max_depth"]),
                "learning_rate": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["learning_rate"]),
                "regularization": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["regularization"]),
                "subsample": optuna.distributions.CategoricalDistribution(SEARCH_SPACE_CATALOG["subsample"]),
                "min_samples_leaf": optuna.distributions.CategoricalDistribution(
                    SEARCH_SPACE_CATALOG["min_samples_leaf"]
                ),
            }
            baseline_params = latest.config.model_dump()
            # Snap baseline params to nearest catalog values if needed
            for k, dist in distributions.items():
                choices = list(dist.choices)
                if baseline_params[k] not in choices:
                    baseline_params[k] = choices[0]

            frozen = optuna.trial.create_trial(
                params=baseline_params,
                distributions=distributions,
                value=latest.diagnostics.val_macro_f1,
            )
            self.study.add_trial(frozen)

        self._last_reported_idx = latest.trial_index

    def suggest_next(self, history: List[TrialRecord]) -> PipelineConfig:
        """Suggest next configuration using Optuna ask/tell interface."""
        self._sync_history(history)

        if not history:
            cfg = PipelineConfig.default_baseline()
            self.seen_hashes.add(cfg.config_hash())
            return cfg

        th_choices = SEARCH_SPACE_CATALOG["threshold_tuning"] if self.n_classes == 2 else [False]

        for _ in range(30):
            trial = self.study.ask()
            cfg = PipelineConfig(
                imputer=trial.suggest_categorical("imputer", SEARCH_SPACE_CATALOG["imputer"]),
                scaler=trial.suggest_categorical("scaler", SEARCH_SPACE_CATALOG["scaler"]),
                categorical_encoder=trial.suggest_categorical(
                    "categorical_encoder", SEARCH_SPACE_CATALOG["categorical_encoder"]
                ),
                feature_engineering=trial.suggest_categorical(
                    "feature_engineering", SEARCH_SPACE_CATALOG["feature_engineering"]
                ),
                imbalance_strategy=trial.suggest_categorical(
                    "imbalance_strategy", SEARCH_SPACE_CATALOG["imbalance_strategy"]
                ),
                threshold_tuning=trial.suggest_categorical("threshold_tuning", th_choices),
                model_family=trial.suggest_categorical("model_family", SEARCH_SPACE_CATALOG["model_family"]),
                n_estimators=trial.suggest_categorical("n_estimators", SEARCH_SPACE_CATALOG["n_estimators"]),
                max_depth=trial.suggest_categorical("max_depth", SEARCH_SPACE_CATALOG["max_depth"]),
                learning_rate=trial.suggest_categorical("learning_rate", SEARCH_SPACE_CATALOG["learning_rate"]),
                regularization=trial.suggest_categorical("regularization", SEARCH_SPACE_CATALOG["regularization"]),
                subsample=trial.suggest_categorical("subsample", SEARCH_SPACE_CATALOG["subsample"]),
                min_samples_leaf=trial.suggest_categorical("min_samples_leaf", SEARCH_SPACE_CATALOG["min_samples_leaf"]),
            )
            h = cfg.config_hash()
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                self._pending_trial = trial
                return cfg
            # Penalize duplicate trial slightly so sampler moves away
            self.study.tell(trial, history[-1].diagnostics.val_macro_f1 - 0.01)

        fallback = sample_random_config(self.rng, self.seen_hashes, n_classes=self.n_classes)
        self.seen_hashes.add(fallback.config_hash())
        return fallback
