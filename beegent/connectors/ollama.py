"""Ollama - the default. Local, no key, no cost.

Talks to Ollama's OpenAI-compatible endpoint, so it inherits everything from
OpenAICompatConnector and only declares what is different: it accepts JSON mode.
"""

import os
from typing import Callable

from beegent.connectors.base import OpenAICompatConnector

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaConnector(OpenAICompatConnector):
    provider = "ollama"
    supports_response_format = True  # unlike hosted Claude endpoints, Ollama accepts it

    # Sized for a local box (RTX 4070 Laptop, 8GB).
    DEFAULT_MODELS = {
        "planner": "deepseek-r1:14b",
        # qwen3 has the most reliable tool calling under Ollama; deepseek-r1 does not.
        # 8b (~5.2GB) fits an 8GB card with room left for a 16k context. Do NOT "upgrade"
        # this to qwen3:14b - it was tried and measured at ~10GB resident, 39% spilled to
        # CPU, and it blew past LLM_TIMEOUT on a single call.
        "geofetch": "qwen3:8b",
        "critic": "deepseek-r1:14b",
    }

    def __init__(self, log: Callable[[str], None] = print) -> None:
        super().__init__(
            base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            api_key="ollama",  # placeholder - Ollama ignores it entirely
            log=log,
        )

    def validate(self) -> str | None:
        """Reachability, not credentials - there are none.

        A pulled-model check is deliberately not done here: it would need Ollama's native
        /api/tags rather than the OpenAI surface, and a missing model already fails loudly
        on the first call with a clear message.
        """
        import requests

        root = self.base_url.rsplit("/v1", 1)[0]
        try:
            requests.get(f"{root}/api/tags", timeout=5).raise_for_status()
        except Exception as exc:
            return (f"cannot reach Ollama at {self.base_url} ({type(exc).__name__}). "
                    "Is it running?  ->  ollama serve")
        return None
