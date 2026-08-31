"""Mistral - hosted, OpenAI-shaped, key-only."""

import logging
import os
from typing import Callable

from beegent.connectors.base import LLMConnector, OpenAICompatConnector

_log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
MODELS_DOC = "https://docs.mistral.ai/getting-started/models/models_overview/"


class MistralConnector(OpenAICompatConnector):
    provider = "mistral"
    supports_response_format = True  # json_object; the planner/critic prompts already say JSON

    # One model for all three roles
    DEFAULT_MODELS = dict.fromkeys(LLMConnector.ROLES, "mistral-small-latest")

    def __init__(self, log: Callable[[str], None] = _log.info) -> None:
        super().__init__(
            base_url=os.environ.get("MISTRAL_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("MISTRAL_API_KEY", ""),
            log=log,
        )

    # No _create() override: Mistral accepts temperature and response_format as sent.
    # No _token_usage() override: it spells usage prompt_tokens/completion_tokens already.
    # Mistral's <think> blocks are handled by llm.strip_think(), which needs no change.

    def validate(self) -> str | None:
        """Key, reachability, and that each role's model is still served."""
        import requests

        if not self.api_key:
            return "mistral backend needs MISTRAL_API_KEY (see .env.example)."
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            served = {m.get("id") for m in resp.json().get("data", [])}
        except Exception as exc:
            return (f"cannot reach Mistral at {self.base_url} ({type(exc).__name__}). "
                    "Check MISTRAL_API_KEY and your connection.")
        # /v1/models catches a typo or a retired id. It does NOT reflect entitlement: it
        # lists models a free key gets 403 tier_not_allowed on, so a run can still fail later.
        missing = sorted({self.model_for(r) for r in self.ROLES} - served)
        if missing:
            # The served list is what makes this actionable; never truncate it.
            return (f"mistral does not list {', '.join(missing)}; "
                    f"listed: {', '.join(sorted(served))}. "
                    f"Set <ROLE>_MODEL or see {MODELS_DOC}.")
        return None
