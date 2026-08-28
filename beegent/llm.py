"""Shared LLM helpers, and the one selected connector."""

import json
import re

from beegent import config
from beegent.connectors import LLMConnector, create_connector
from beegent.schemas import TokenUsage

#: Built on first USE, not first import, so run.py's --backend flag still decides.
_connector: LLMConnector | None = None


def get_connector() -> LLMConnector:
    """The selected backend. Built once, on first use."""
    global _connector
    if _connector is None:
        _connector = create_connector(config.LLM_BACKEND)
    return _connector


def select_backend(provider: str) -> LLMConnector:
    """Override the environment's choice of backend."""
    global _connector
    _connector = create_connector(provider)
    return _connector


#: Per-role token spend; planner and critic produce no Candidate to hang a cost on.
_usage: dict[str, TokenUsage] = {}


def reset_usage() -> None:
    """Clear the meter. beegent/run.py:discover() calls this once at the start of a run."""
    _usage.clear()


def usage_by_role() -> dict[str, TokenUsage]:
    """A copy of what each role has spent since the last reset_usage()."""
    return dict(_usage)


def message_text(msg) -> str:
    """Flatten an assistant message's content to text."""
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
    """Remove <think>...</think> reasoning blocks (deepseek-r1, qwen3, ...)."""
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


def chat_json(role: str, messages: list[dict]) -> tuple:
    """One JSON-mode completion, retried once -> (data, TokenUsage); unpack before testing."""
    c = get_connector()
    data, usage = c.chat_json(c.model_for(role), messages)
    _usage.setdefault(role, TokenUsage()).add(usage)
    return data, usage


def chat_tools(role: str, messages: list[dict], tools: list[dict]) -> tuple:
    """One tool-calling completion -> (message, TokenUsage)."""
    c = get_connector()
    return c.chat_tools(c.model_for(role), messages, tools)


def to_message_dict(msg) -> dict:
    """Serialize an assistant message for the next request."""
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
