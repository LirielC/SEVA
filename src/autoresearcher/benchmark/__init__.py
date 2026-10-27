"""Benchmark orchestration and comparative report generation."""

from autoresearcher.benchmark.runner import (
    generate_markdown_report,
    run_full_benchmark,
    run_single_strategy,
)

__all__ = ["run_single_strategy", "run_full_benchmark", "generate_markdown_report"]
