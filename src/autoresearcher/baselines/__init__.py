"""Competing optimization baselines operating over the shared 5-layer CASH search space."""

from autoresearcher.baselines.grid_search import GridSearchStrategy
from autoresearcher.baselines.optuna_search import OptunaTPEStrategy
from autoresearcher.baselines.random_search import RandomSearchStrategy
from autoresearcher.baselines.space_catalog import SEARCH_SPACE_CATALOG, sample_random_config

__all__ = [
    "SEARCH_SPACE_CATALOG",
    "sample_random_config",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "OptunaTPEStrategy",
]
