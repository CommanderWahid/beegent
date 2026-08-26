"""Shared LLM helpers, and the one selected connector.

The backend itself lives in `beegent/connectors/` - this module holds only what is genuinely
backend-agnostic (response-shape flattening, <think> stripping, JSON extraction) plus two
thin delegations so every caller keeps importing `chat_json` / `chat_tools` from here.

Why a connector layer at all: backend quirks used to live in this file behind
`if config.LLM_BACKEND == ...` branches, in a module whose own docstring claimed there was
"one client and two base_urls, not two code paths". Each backend now owns its quirks, and a
new one - including a non-OpenAI-shaped one - is a subclass plus a registry entry.
"""

import json
import re

from beegent import config
from beegent.connectors import LLMConnector, create_connector

#: Built on first USE, not first import. That is what lets run.py's --backend flag decide
#: before anything is constructed: by the time main() parses arguments, this module has long
#: been imported, so an eager connector would already have picked the env-derived backend.
_connector: LLMConnector | None = None


def get_connector() -> LLMConnector:
    """The selected backend. Built once, on first use."""
    global _connector
    if _connector is None:
        _connector = create_connector(config.LLM_BACKEND)
    return _connector


def select_backend(provider: str) -> LLMConnector:
    """Override the environment's choice of backend. run.py calls this from --backend.

    Raises ValueError naming the valid providers if `provider` is not registered, so a typo
    fails at startup rather than as a confusing failure on the first LLM call.
    """
    global _connector
    _connector = create_connector(provider)
    return _connector


def message_text(msg) -> str:
    """Flatten an assistant message's content to text.

    Not a backend branch despite its origin: most endpoints return a plain string, but some
    return a list of content blocks. Normalizing here keeps that shape from leaking into
    parse_json() and every caller downstream, whichever connector produced it.
    """
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (deepseek-r1, qwen3, ...).

    Both Ollama defaults are reasoning models, so this is on the hot path, not an edge case.
    """
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def parse_json(raw: str) -> dict | None:
    """Parse a JSON object out of a completion, tolerating fences and prose around it."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Some models wrap JSON in prose or fences - grab the outermost object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def chat_json(role: str, messages: list[dict]) -> dict | None:
    """One JSON-mode completion, retried once. None if no attempt produced JSON.

    `role` is a pipeline stage ("planner" / "critic"), not a model name: the connector maps
    it to a model, so which model a stage uses is the backend's business, not the caller's.

    Empty completions do happen - a thinking model under GPU pressure can return nothing at
    all - and observed failures were transient, so a retry usually clears them.

    Callers must treat None as "no answer", never as a negative answer.
    """
    c = get_connector()
    return c.chat_json(c.model_for(role), messages)


def chat_tools(role: str, messages: list[dict], tools: list[dict]) -> tuple:
    """One tool-calling completion. Returns (assistant_message, TokenUsage).

    `role` is a pipeline stage ("geofetch"), not a model name - see chat_json above.

    The usage half is what gives a run its cost meter - the geofetch agent sums it across
    every step and beegent/run.py reports the total. Backends that report no usage yield a
    zeroed TokenUsage rather than None, so callers never branch on it.
    """
    c = get_connector()
    return c.chat_tools(c.model_for(role), messages, tools)


def to_message_dict(msg) -> dict:
    """
    Serialize an assistant message for the next request.

    Drops Ollama's non-standard `reasoning` field, and strips <think> blocks from the
    content, so thinking models stay swappable without touching the agents. Both matter for
    kept history specifically: a reasoning block is large, useful only in the moment, and
    left in place it crowds out the system prompt on a small context window.
    """
    out: dict = {"role": "assistant", "content": strip_think(message_text(msg))}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out
