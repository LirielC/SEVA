"""Global settings and environment configuration for AutoResearcher."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    llm_base_url: str = os.getenv("AUTORESEARCHER_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str = os.getenv("AUTORESEARCHER_LLM_API_KEY", "")
    llm_model: str = os.getenv("AUTORESEARCHER_LLM_MODEL", "qwen/qwen-2.5-7b-instruct")
    agent_mode: str = os.getenv("AUTORESEARCHER_AGENT_MODE", "auto")
    artifacts_dir: Path = Path(os.getenv("AUTORESEARCHER_ARTIFACTS_DIR", "artifacts"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow.db")

    @property
    def has_valid_llm_key(self) -> bool:
        """Check whether an external LLM key or local Ollama endpoint is configured."""
        if "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url:
            return bool(self.llm_api_key)
        return bool(self.llm_api_key and self.llm_api_key != "your_api_key_here")


DEFAULT_BENCHMARK_DATASETS = [
    "credit-g",
    "phoneme",
    "vehicle",
]

SETTINGS = Settings()
