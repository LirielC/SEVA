"""Uniform Random Search baseline over the 5-layer CASH search space."""

from __future__ import annotations

from typing import List, Set
import numpy as np

from autoresearcher.baselines.space_catalog import sample_random_config
from autoresearcher.schemas import PipelineConfig, TrialRecord


class RandomSearchStrategy:
    """Samples configurations uniformly at random after running the shared Trial 0 baseline."""

    name: str = "Random Search"

    def __init__(self, seed: int = 42, n_classes: int = 2) -> None:
        self.seed = seed
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.seen_hashes: Set[str] = set()

    def suggest_next(self, history: List[TrialRecord]) -> PipelineConfig:
        """Return default baseline on Trial 0, then uniform random unique configs."""
        if not history:
            cfg = PipelineConfig.default_baseline()
            self.seen_hashes.add(cfg.config_hash())
            return cfg

        cfg = sample_random_config(self.rng, self.seen_hashes, n_classes=self.n_classes)
        self.seen_hashes.add(cfg.config_hash())
        return cfg
