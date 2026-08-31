"""What beegent needs from an LLM backend - nothing more."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Callable

from beegent import config
from beegent.schemas import TokenUsage

_log = logging.getLogger(__name__)


class LLMConnector(ABC):
    """The contract. `provider` is the key this registers under in CONNECTORS."""

    provider: str = "abstract"

    #: Pipeline stages that call an LLM; each resolves to a model independently.
    ROLES: tuple[str, ...] = ("planner", "geofetch", "critic")

    #: role -> model name for THIS backend; every connector declares its own.
    DEFAULT_MODELS: dict[str, str] = {}

    def model_for(self, role: str) -> str:
        """Resolve a pipeline role to a model name for this backend."""
        if role not in self.ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {list(self.ROLES)}")
        override = os.environ.get(f"{role.upper()}_MODEL")
        if override:
            return override
        try:
            return self.DEFAULT_MODELS[role]
        except KeyError:
            raise ValueError(
                f"{type(self).__name__} declares no default model for role {role!r}; "
                f"add it to DEFAULT_MODELS or set {role.upper()}_MODEL"
            ) from None

    @abstractmethod
    def chat_json(self, model: str, messages: list[dict]) -> tuple:
        """One JSON-mode completion -> (data, TokenUsage); data None means no answer."""

    @abstractmethod
    def chat_tools(self, model: str, messages: list[dict], tools: list[dict]) -> tuple:
        """One tool-calling completion -> (message, TokenUsage)."""

    @abstractmethod
    def validate(self) -> str | None:
        """Check credentials and reachability BEFORE a run spends steps."""


class OpenAICompatConnector(LLMConnector):
    """Shared implementation for every backend speaking the OpenAI protocol."""

    #: response_format JSON mode; Claude serving endpoints 400 on it, so subclasses opt in.
    supports_response_format: bool = False

    def __init__(self, base_url: str, api_key: str,
                 log: Callable[[str], None] = _log.info) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.log = log
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # lazy: tests never construct a real client

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=config.LLM_TIMEOUT,
                max_retries=config.LLM_MAX_RETRIES,
            )
        return self._client

    def _create(self, model: str, messages: list[dict], **kwargs):
        """One completion. Subclasses override to handle their own parameter quirks."""
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    def _token_usage(self, raw) -> TokenUsage:
        """Map ONE response's raw usage payload to TokenUsage."""
        # Cache hits are nested one level down; a backend not reporting them leaves this 0.
        details = getattr(raw, "prompt_tokens_details", None)
        cached = (details.get("cached_tokens") if isinstance(details, dict)
                  else getattr(details, "cached_tokens", 0))
        return TokenUsage(
            prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
            cached_tokens=cached or 0,
        )

    def chat_json(self, model: str, messages: list[dict]) -> tuple:
        from beegent.llm import message_text, parse_json

        spent = TokenUsage()
        kwargs = {"response_format": {"type": "json_object"}} if self.supports_response_format else {}
        for attempt in range(config.CHAT_JSON_ATTEMPTS):
            # Off 0 on retry: a temp-0 replay would reproduce the same bad completion.
            resp = self._create(model, messages,
                                temperature=0 if attempt == 0 else 0.3, **kwargs)
            # Charged before the parse: a discarded attempt still cost tokens.
            spent.add(self._token_usage(getattr(resp, "usage", None)))
            raw = message_text(resp.choices[0].message)
            parsed = parse_json(raw)
            if parsed is not None:
                return parsed, spent
            self.log(f"  [llm] {model} returned non-JSON "
                     f"(attempt {attempt + 1}/{config.CHAT_JSON_ATTEMPTS}): {raw[:120]!r}")
        return None, spent

    def chat_tools(self, model: str, messages: list[dict], tools: list[dict]) -> tuple:
        resp = self._create(model, messages, tools=tools, temperature=0)
        return resp.choices[0].message, self._token_usage(getattr(resp, "usage", None))
