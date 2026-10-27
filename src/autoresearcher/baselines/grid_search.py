"""Systematic Grid Search baseline over a low-discrepancy design of the 5-layer search space."""

from __future__ import annotations

import itertools
from typing import List, Set

from autoresearcher.schemas import PipelineConfig, TrialRecord


class GridSearchStrategy:
    """Evaluates a deterministic, coarse orthogonal grid across models, encoders, and regularizers."""

    name: str = "Grid Search"

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes
        self.seen_hashes: Set[str] = set()
        self.grid_queue: List[PipelineConfig] = self._build_orthogonal_grid()
        self.cursor: int = 0

    def _build_orthogonal_grid(self) -> List[PipelineConfig]:
        """Construct a systematic cartesian grid ordered from simple to complex."""
        models = ["random_forest", "xgboost", "catboost"]
        scalers = ["none", "standard"]
        encoders = ["onehot", "ordinal"]
        imbalances = ["none", "class_weight_balanced"]
        depths = [10, 6, 4]
        fe_ops = ["none", "mutual_info_k80", "poly_top5"]

        configs: List[PipelineConfig] = [PipelineConfig.default_baseline()]
        for fe, imb, depth, model, enc, sc in itertools.product(
            fe_ops, imbalances, depths, models, encoders, scalers
        ):
            cfg = PipelineConfig(
                imputer="median",
                scaler=sc,  # type: ignore[arg-type]
                categorical_encoder=enc,  # type: ignore[arg-type]
                feature_engineering=fe,  # type: ignore[arg-type]
                imbalance_strategy=imb,  # type: ignore[arg-type]
                threshold_tuning=False,
                model_family=model,  # type: ignore[arg-type]
                n_estimators=150,
                max_depth=depth,
                learning_rate=0.1,
                regularization=1.0,
                subsample=0.9,
                min_samples_leaf=2,
            )
            configs.append(cfg)
        return configs

    def suggest_next(self, history: List[TrialRecord]) -> PipelineConfig:
        """Return next unvisited grid point."""
        while self.cursor < len(self.grid_queue):
            cfg = self.grid_queue[self.cursor]
            self.cursor += 1
            h = cfg.config_hash()
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                return cfg
        return PipelineConfig.default_baseline()
