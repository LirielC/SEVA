"""Declarative leak-free ML pipeline builder and cross-validation diagnostic evaluator."""

from autoresearcher.pipeline.builder import build_preprocessor_and_fe, instantiate_classifier
from autoresearcher.pipeline.evaluator import evaluate_on_holdout, evaluate_pipeline_cv

__all__ = [
    "build_preprocessor_and_fe",
    "instantiate_classifier",
    "evaluate_pipeline_cv",
    "evaluate_on_holdout",
]
