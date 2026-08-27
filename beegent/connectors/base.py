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
    def chat_json(self, model: str, messages: list[dict]) -> tuple:
        """One JSON-mode completion. Returns (data, TokenUsage) - same shape as chat_tools.

        The annotation is a bare `tuple` to match chat_tools, whose first half is an SDK
        object with no useful static type, so the shape lives here rather than in the
        signature.

        `data` of None means "no answer", never a negative answer. Callers must UNPACK
        before testing it: a 2-tuple is always truthy, so `if chat_json(...)` reads a None
        answer as an answer.

        The usage half must cover EVERY request the call made, including attempts whose
        output was discarded as unparseable - a connector that retries internally is the
        only layer that can see them, which is why this returns usage at all. Backends that
        report no usage return a zeroed TokenUsage rather than None.
        """

    @abstractmethod
    def chat_tools(self, model: str, messages: list[dict], tools: list[dict]) -> tuple:
        """One tool-calling completion. Returns (assistant_message, TokenUsage).

        Backends that report no usage must return a zeroed TokenUsage rather than None, so
        callers never have to branch on it.
        """

    @abstractmethod
    def validate(self) -> str | None:
        """Check credentials and reachability BEFORE a run spends steps.

        Returns None when ready, or the error message to show the user.

        Abstract on purpose, with no permissive default: every backend reached over a wire
        has SOMETHING worth checking first, and a connector that silently inherited a
        `return None` would look validated while checking nothing. A backend that genuinely
        has nothing to check says so explicitly with `return None` of its own.
        """


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

    def _token_usage(self, raw) -> TokenUsage:
        """Map ONE response's raw usage payload to TokenUsage. The single place any token
        number is read, so both chat_json and chat_tools report on identical terms.

        Isolated and overridable because these field names are a provider convention, not a
        standard - `prompt_tokens`/`completion_tokens` is OpenAI's spelling, and a backend
        using `input_tokens`/`output_tokens` needs only this method. It is also the seam for
        metrics beyond these two (cache hits, reasoning tokens): they get added here and to
        TokenUsage, and connectors that report them override this to fill them in.

        Never raises. An unfamiliar shape degrades to zeros rather than killing a run - but
        note that a zero is then ambiguous between "not reported" and "not read", and only a
        raw usage object from a live endpoint tells those apart. Do not guess field names
        for a provider you have not observed.
        """
        return TokenUsage(
            prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        )

    def chat_json(self, model: str, messages: list[dict]) -> tuple:
        from beegent.llm import message_text, parse_json

        spent = TokenUsage()
        kwargs = {"response_format": {"type": "json_object"}} if self.supports_response_format else {}
        for attempt in range(config.CHAT_JSON_ATTEMPTS):
            # Retry nudges temperature off 0: a temp-0 replay is byte-identical and would
            # just reproduce a deterministically-bad completion.
            resp = self._create(model, messages,
                                temperature=0 if attempt == 0 else 0.3, **kwargs)
            # Charged BEFORE the parse decides anything: a discarded non-JSON attempt cost
            # real tokens, and leaving it out is what made an internal retry look free.
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
