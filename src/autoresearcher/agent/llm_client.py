"""OpenAI-compatible LLM client with token tracking, cost telemetry, and schema repair."""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional
from openai import OpenAI

from autoresearcher.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from autoresearcher.baselines.space_catalog import SEARCH_SPACE_CATALOG
from autoresearcher.config import SETTINGS, Settings
from autoresearcher.schemas import AgentDecision, DatasetProfile, PipelineConfig, TokenUsage, TrialRecord

# Default pricing estimate for Qwen2.5-7B on typical inference providers (~$0.20 per 1M tokens)
DEFAULT_COST_PER_1M_INPUT = 0.15
DEFAULT_COST_PER_1M_OUTPUT = 0.25


def _extract_json_block(text: str) -> str:
    """Extract JSON object from raw LLM response even if wrapped in markdown code fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    return text


def _sanitize_and_repair_config(raw_cfg: dict, n_classes: int) -> PipelineConfig:
    """Ensure LLM-suggested fields conform strictly to valid catalog domains."""
    sanitized = raw_cfg.copy()

    # Snap categoricals to nearest valid catalog choice if corrupted
    for cat_key in ["imputer", "scaler", "categorical_encoder", "feature_engineering", "imbalance_strategy", "model_family"]:
        valid_choices = SEARCH_SPACE_CATALOG[cat_key]
        if sanitized.get(cat_key) not in valid_choices:
            sanitized[cat_key] = valid_choices[0]

    # Numeric constraints
    sanitized["n_estimators"] = int(np_clip(sanitized.get("n_estimators", 150), 50, 500))
    sanitized["max_depth"] = int(np_clip(sanitized.get("max_depth", 6), 3, 12))
    sanitized["learning_rate"] = float(np_clip(sanitized.get("learning_rate", 0.1), 0.01, 0.30))
    sanitized["regularization"] = float(np_clip(sanitized.get("regularization", 1.0), 0.01, 20.0))
    sanitized["subsample"] = float(np_clip(sanitized.get("subsample", 0.9), 0.5, 1.0))
    sanitized["min_samples_leaf"] = int(np_clip(sanitized.get("min_samples_leaf", 2), 1, 20))

    if n_classes > 2 or not isinstance(sanitized.get("threshold_tuning"), bool):
        sanitized["threshold_tuning"] = False

    return PipelineConfig.model_validate(sanitized)


def np_clip(val: object, low: float, high: float) -> float:
    try:
        f = float(val)  # type: ignore[arg-type]
        return max(low, min(high, f))
    except Exception:
        return (low + high) / 2.0


class LLMResearcherClient:
    """Client that calls an OpenAI-compatible chat completions endpoint and tracks telemetry."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._client: Optional[OpenAI] = None
        if settings.has_valid_llm_key:
            self._client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=25.0,
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None and self.settings.agent_mode != "diagnostic_heuristic"

    def request_decision(
        self,
        profile: DatasetProfile,
        history: List[TrialRecord],
        budget: int,
    ) -> Optional[AgentDecision]:
        """Query the configured LLM for the next AgentDecision with telemetry; return None on failure."""
        if not self.is_available or self._client is None:
            return None

        user_prompt = build_user_prompt(profile=profile, history=history, budget=budget)
        start_t = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.25,
                max_tokens=900,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            # Telemetry extraction
            usage = getattr(response, "usage", None)
            p_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            c_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            tot_tokens = int(getattr(usage, "total_tokens", p_tokens + c_tokens) or (p_tokens + c_tokens))

            cost = (p_tokens / 1_000_000.0) * DEFAULT_COST_PER_1M_INPUT + (
                c_tokens / 1_000_000.0
            ) * DEFAULT_COST_PER_1M_OUTPUT

            telemetry = TokenUsage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=tot_tokens,
                estimated_cost_usd=round(cost, 6),
                call_latency_ms=round(elapsed_ms, 2),
            )

            content = response.choices[0].message.content or ""
            json_str = _extract_json_block(content)
            parsed = json.loads(json_str)

            # Strict self-repair for config
            raw_cfg = parsed.get("config", {})
            repaired_cfg = _sanitize_and_repair_config(raw_cfg, n_classes=profile.n_classes)

            decision = AgentDecision(
                bottleneck_diagnosis=str(parsed.get("bottleneck_diagnosis", "Diagnostic error pattern")),
                scientific_hypothesis=str(parsed.get("scientific_hypothesis", "Intervention hypothesis")),
                expected_outcome=str(
                    parsed.get(
                        "expected_outcome",
                        "Improvement in validation Macro-F1 with mitigated bottleneck pathology",
                    )
                ),
                exploration_mode=parsed.get("exploration_mode", "exploit_refinement"),
                changed_components=parsed.get("changed_components", []),
                config=repaired_cfg,
                token_usage=telemetry,
            )
            return decision

        except Exception:
            return None
