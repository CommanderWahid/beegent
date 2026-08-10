"""One OpenAI-SDK client for every LLM call, with the backend as a config switch.

Both supported backends speak the OpenAI wire format: Ollama exposes it at
/v1, and Databricks Model Serving / Unity AI Gateway exposes it at
/serving-endpoints. So there is one client and two base_urls, not two code paths.
"""

import json

from openai import OpenAI

import config


def get_llm_client(backend: str = config.LLM_BACKEND) -> OpenAI:
    if backend == "ollama":
        return OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # placeholder, unused by Ollama
            timeout=config.LLM_TIMEOUT,
        )
    if backend == "databricks":
        # TODO: PAT for now; move to OAuth machine-to-machine before production.
        return OpenAI(
            base_url=f"https://{config.DATABRICKS_HOST}/serving-endpoints",
            api_key=config.DATABRICKS_TOKEN,
            timeout=config.LLM_TIMEOUT,
        )
    raise ValueError(f"unknown LLM_BACKEND: {backend!r}")


client = get_llm_client()


def chat_json(model: str, messages: list[dict]) -> dict | None:
    """One JSON-mode completion. Returns None if the model didn't produce JSON.

    POC shortcut: callers fall back to a deterministic default rather than retry.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose or fences - grab the outermost object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        print(f"  [llm] {model} returned non-JSON: {raw[:120]!r}")
        return None


def chat_tools(model: str, messages: list[dict], tools: list[dict]):
    """One tool-calling completion. Returns the assistant message."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0,
    )
    return resp.choices[0].message


def to_message_dict(msg) -> dict:
    """Serialize an assistant message for the next request.

    Drops Ollama's non-standard `reasoning` field so thinking models stay
    swappable without touching the agents.
    """
    out: dict = {"role": "assistant", "content": msg.content or ""}
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
