"""Groq - hosted, OpenAI-shaped, key-only."""

import logging
import os
from typing import Callable

from beegent.connectors.base import LLMConnector, OpenAICompatConnector

_log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MODELS_DOC = "https://console.groq.com/docs/models"


class GroqConnector(OpenAICompatConnector):
    provider = "groq"
    supports_response_format = True  # json_object; the planner/critic prompts already say JSON

    # One model for all three roles: Groq's production chat tier is two models wide, and
    # gpt-oss-20b is a cheaper GEOFETCH_MODEL override rather than a default worth shipping.
    DEFAULT_MODELS = dict.fromkeys(LLMConnector.ROLES, "openai/gpt-oss-120b")

    def __init__(self, log: Callable[[str], None] = _log.info) -> None:
        super().__init__(
            base_url=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("GROQ_API_KEY", ""),
            log=log,
        )

    # No _create() override: unlike Databricks, Groq accepts temperature (it coerces 0 to 1e-8).
    # No _token_usage() override: it spells usage prompt_tokens/completion_tokens already.

    def validate(self) -> str | None:
        """Key, reachability, and that each role's model is still served."""
        import requests

        if not self.api_key:
            return "groq backend needs GROQ_API_KEY (see .env.example)."
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            served = {m.get("id") for m in resp.json().get("data", [])}
        except Exception as exc:
            return (f"cannot reach Groq at {self.base_url} ({type(exc).__name__}). "
                    "Check GROQ_API_KEY and your connection.")
        # Groq retires model ids; catch it here rather than as a 404 mid-angle.
        missing = sorted({self.model_for(r) for r in self.ROLES} - served)
        if missing:
            # The served list is what makes this actionable; never truncate it.
            return (f"groq no longer serves {', '.join(missing)}; "
                    f"served: {', '.join(sorted(served))}. "
                    f"Set <ROLE>_MODEL or see {MODELS_DOC}.")
        return None
