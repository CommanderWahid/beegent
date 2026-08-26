"""What beegent needs from an LLM backend - nothing more.

Two call shapes, because the pipeline genuinely uses two: `chat_json()` for the planner and
critic (one shot, JSON out) and `chat_tools()` for geofetch (function calling, in a loop).

`OpenAICompatConnector` implements both once for any backend speaking the OpenAI wire
format, so adding one of those is a subclass with a base_url and an auth header. A backend
with a different wire format subclasses `LLMConnector` directly: mapping its response into
`(message, TokenUsage)` is the whole integration job.

The point of the layer is not only extensibility. Backend quirks used to live in shared
code - a temperature retry that existed for one Databricks endpoint, a `response_format`
that only Ollama accepts - so a module claiming to be backend-agnostic carried three
`if backend == ...` branches. Each connector now owns its own quirks.
"""

import os
from abc import ABC, abstractmethod
from typing import Callable

from beegent import config
from beegent.schemas import TokenUsage


class LLMConnector(ABC):
    """The contract. `provider` is the key this registers under in CONNECTORS."""

    provider: str = "abstract"

    #: The pipeline stages that call an LLM. Each resolves to a model independently, so a
    #: cheap model can drive the expensive loop while a strong one judges the result.
    ROLES: tuple[str, ...] = ("planner", "geofetch", "critic")

    #: role -> model name for THIS backend. Every connector declares its own; nothing
    #: outside this package needs to know what they are.
    DEFAULT_MODELS: dict[str, str] = {}

    def model_for(self, role: str) -> str:
        """Resolve a pipeline role to a model name for this backend.

        An env var named after the role wins - PLANNER_MODEL, GEOFETCH_MODEL, CRITIC_MODEL -
        so a single step can be pointed at a different model without touching code.

        Raises on an unknown role rather than returning None: a typo must fail here, at the
        call, not become a request against an empty model name that fails obscurely later.
        """
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
    def chat_json(self, model: str, messages: list[dict]) -> dict | None:
        """One JSON-mode completion. None means "no answer", never a negative answer."""

    @abstractmethod
    def chat_tools(self, model: str, messages: list[dict], tools: list[dict]) -> tuple:
        """One tool-calling completion. Returns (assistant_message, TokenUsage).

        Backends that report no usage must return a zeroed TokenUsage rather than None, so
        callers never have to branch on it.
        """

    def validate(self) -> str | None:
        """Check credentials and reachability BEFORE a run spends steps.

        Returns None when ready, or the error message to show the user. Default is
        permissive: a backend with nothing to check simply passes.
        """
        return None


class OpenAICompatConnector(LLMConnector):
    """Shared implementation for every backend speaking the OpenAI protocol."""

    #: JSON mode via response_format. Ollama supports it; Databricks' Claude serving
    #: endpoints 400 on it (INVALID_PARAMETER_VALUE). Subclasses opt in.
    supports_response_format: bool = False

    def __init__(self, base_url: str, api_key: str,
                 log: Callable[[str], None] = print) -> None:
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

    def chat_json(self, model: str, messages: list[dict]) -> dict | None:
        from beegent.llm import message_text, parse_json

        kwargs = {"response_format": {"type": "json_object"}} if self.supports_response_format else {}
        for attempt in range(config.CHAT_JSON_ATTEMPTS):
            # Retry nudges temperature off 0: a temp-0 replay is byte-identical and would
            # just reproduce a deterministically-bad completion.
            resp = self._create(model, messages,
                                temperature=0 if attempt == 0 else 0.3, **kwargs)
            raw = message_text(resp.choices[0].message)
            parsed = parse_json(raw)
            if parsed is not None:
                return parsed
            self.log(f"  [llm] {model} returned non-JSON "
                     f"(attempt {attempt + 1}/{config.CHAT_JSON_ATTEMPTS}): {raw[:120]!r}")
        return None

    def chat_tools(self, model: str, messages: list[dict], tools: list[dict]) -> tuple:
        resp = self._create(model, messages, tools=tools, temperature=0)
        raw = getattr(resp, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        )
        return resp.choices[0].message, usage
