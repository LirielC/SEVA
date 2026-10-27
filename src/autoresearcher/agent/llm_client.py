"""OpenAI-compatible LLM client for querying Qwen2.5-Instruct or other research models."""

from __future__ import annotations

import json
import re
from typing import List, Optional
from openai import OpenAI

from autoresearcher.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from autoresearcher.config import SETTINGS, Settings
from autoresearcher.schemas import AgentDecision, DatasetProfile, TrialRecord


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


class LLMResearcherClient:
    """Client that calls an OpenAI-compatible chat completions endpoint and validates AgentDecision."""

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
        """Query the configured LLM for the next AgentDecision; return None on network/parse error."""
        if not self.is_available or self._client is None:
            return None

        user_prompt = build_user_prompt(profile=profile, history=history, budget=budget)
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
            content = response.choices[0].message.content or ""
            json_str = _extract_json_block(content)
            parsed = json.loads(json_str)
            decision = AgentDecision.model_validate(parsed)
            # Enforce binary-only constraint for threshold_tuning
            if profile.n_classes > 2 and decision.config.threshold_tuning:
                decision.config.threshold_tuning = False
            return decision
        except Exception:
            return None
