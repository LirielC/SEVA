"""Unit tests for the 5-layer PipelineConfig search space, hashing, and diff detection."""

from __future__ import annotations

import numpy as np

from autoresearcher.baselines.space_catalog import sample_random_config
from autoresearcher.schemas import PipelineConfig


def test_default_baseline_and_canonical_hash() -> None:
    base1 = PipelineConfig.default_baseline()
    base2 = PipelineConfig.default_baseline()
    assert base1.config_hash() == base2.config_hash()

    # Changing learning_rate on random_forest should NOT change canonical hash
    rf_diff_lr = base1.model_copy(update={"learning_rate": 0.25})
    assert rf_diff_lr.config_hash() == base1.config_hash()

    # Changing max_depth or model_family MUST change canonical hash
    xgb_cfg = base1.model_copy(update={"model_family": "xgboost", "max_depth": 5})
    assert xgb_cfg.config_hash() != base1.config_hash()
    diffs = xgb_cfg.diff_from(base1)
    assert any("model_family" in d for d in diffs)
    assert any("max_depth" in d for d in diffs)


def test_sample_random_config_uniqueness() -> None:
    rng = np.random.default_rng(42)
    seen: set[str] = set()
    for _ in range(25):
        cfg = sample_random_config(rng, seen, n_classes=2)
        h = cfg.config_hash()
        assert h not in seen
        seen.add(h)
    assert len(seen) == 25
