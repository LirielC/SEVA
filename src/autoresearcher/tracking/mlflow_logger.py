"""MLflow logger and persistent JSON/Markdown Lab Notebook writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from autoresearcher.config import SETTINGS
from autoresearcher.schemas import MethodBenchmarkResult, TrialRecord

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


class ExperimentTracker:
    """Logs trials and benchmark summaries to both MLflow and local JSON Lab Notebooks."""

    def __init__(self, artifacts_dir: Optional[Path] = None, enable_mlflow: bool = True) -> None:
        self.artifacts_dir = Path(artifacts_dir or SETTINGS.artifacts_dir)
        self.runs_dir = self.artifacts_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow and HAS_MLFLOW

        if self.enable_mlflow:
            try:
                mlruns_path = (self.artifacts_dir / "mlruns").resolve()
                mlruns_path.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(mlruns_path.as_uri())
                mlflow.set_experiment("AutoResearcher_Benchmark")
            except Exception:
                self.enable_mlflow = False

    def log_method_run(self, result: MethodBenchmarkResult) -> Path:
        """Persist full MethodBenchmarkResult to JSON and optionally log to MLflow."""
        safe_strategy = (
            result.strategy_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        out_file = self.runs_dir / f"{result.dataset_name}__{safe_strategy}__seed{result.seed}.json"
        out_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        if self.enable_mlflow:
            try:
                run_name = f"{result.dataset_name}_{result.strategy_name}_seed{result.seed}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({
                        "dataset": result.dataset_name,
                        "strategy": result.strategy_name,
                        "seed": str(result.seed),
                    })
                    mlflow.log_params({
                        f"best_{k}": str(v) for k, v in result.best_config.model_dump().items()
                    })
                    mlflow.log_metrics({
                        "best_val_macro_f1": result.best_val_macro_f1,
                        "holdout_test_macro_f1": result.holdout_test_macro_f1,
                        "baseline_val_macro_f1": result.baseline_val_macro_f1,
                        "improvement_over_baseline": result.improvement_over_baseline,
                        "trials_to_95pct_peak": float(result.trials_to_95pct_peak),
                        "total_elapsed_seconds": result.total_elapsed_seconds,
                    })
                    for t in result.trials:
                        mlflow.log_metric("trial_val_macro_f1", t.diagnostics.val_macro_f1, step=t.trial_index)
                        mlflow.log_metric("best_val_macro_f1_so_far", t.best_val_macro_f1_so_far, step=t.trial_index)
                        mlflow.log_metric("overfit_gap", t.diagnostics.overfit_gap, step=t.trial_index)
                    mlflow.log_artifact(str(out_file))
            except Exception:
                pass

        return out_file

    def load_all_results(self) -> List[MethodBenchmarkResult]:
        """Load all saved MethodBenchmarkResult JSON files from artifacts/runs."""
        results: List[MethodBenchmarkResult] = []
        if not self.runs_dir.exists():
            return results
        for path in sorted(self.runs_dir.glob("*.json")):
            if path.name == "benchmark_summary.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append(MethodBenchmarkResult.model_validate(data))
            except Exception:
                continue
        return results
