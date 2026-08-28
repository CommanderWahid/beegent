"""Ollama - the default."""

import logging
import os
from typing import Callable

from beegent.connectors.base import OpenAICompatConnector

_log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaConnector(OpenAICompatConnector):
    provider = "ollama"
    supports_response_format = True  # unlike hosted Claude endpoints, Ollama accepts it

    # Sized for a local box (RTX 4070 Laptop, 8GB).
    DEFAULT_MODELS = {
        "planner": "deepseek-r1:14b",
        # qwen3:8b has the most reliable tool calling here; do NOT "upgrade" to 14b.
        "geofetch": "qwen3:8b",
        "critic": "deepseek-r1:14b",
    }

    def __init__(self, log: Callable[[str], None] = _log.info) -> None:
        super().__init__(
            base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            api_key="ollama",  # placeholder - Ollama ignores it entirely
            log=log,
        )

    def validate(self) -> str | None:
        """Reachability, not credentials - there are none."""
        import requests

        root = self.base_url.rsplit("/v1", 1)[0]
        try:
            requests.get(f"{root}/api/tags", timeout=5).raise_for_status()
        except Exception as exc:
            return (f"cannot reach Ollama at {self.base_url} ({type(exc).__name__}). "
                    "Is it running?  ->  ollama serve")
        return None
