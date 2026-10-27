"""Benchmark orchestration, ablation studies, and markdown report generation."""

from autoresearcher.benchmark.runner import (
    generate_markdown_report,
    replay_pipeline_from_config,
    run_full_benchmark,
    run_single_strategy,
)

__all__ = [
    "run_single_strategy",
    "run_full_benchmark",
    "replay_pipeline_from_config",
    "generate_markdown_report",
]
