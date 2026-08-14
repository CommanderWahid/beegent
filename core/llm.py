"""One OpenAI-SDK client for every LLM call, with the backend as a config switch.

Both supported backends speak the OpenAI wire format: Ollama exposes it at
/v1, and Databricks Model Serving / Unity AI Gateway exposes it at
/serving-endpoints. So there is one client and two base_urls, not two code paths.
"""

import json

from openai import BadRequestError, OpenAI

import config


def get_llm_client(backend: str = config.LLM_BACKEND) -> OpenAI:
    if backend == "ollama":
        return OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # placeholder, unused by Ollama
            timeout=config.LLM_TIMEOUT,
        )
    if backend == "databricks":
        # DATABRICKS_HOST may be a bare hostname or the full https://...  URL as shown in
        # the Databricks UI - accept either instead of silently building a malformed,
        # unresolvable base_url out of a pasted-in scheme/trailing slash.
        host = config.DATABRICKS_HOST.removeprefix("https://").removeprefix("http://").rstrip("/")
        # TODO: PAT for now; move to OAuth machine-to-machine before production.
        return OpenAI(
            base_url=f"https://{host}/serving-endpoints",
            api_key=config.DATABRICKS_TOKEN,
            timeout=config.LLM_TIMEOUT,
        )
    raise ValueError(f"unknown LLM_BACKEND: {backend!r}")


client = get_llm_client()


def _create(model: str, messages: list[dict], **kwargs):
    """One completion, retried without `temperature` if the endpoint refuses that parameter.

    Databricks' Claude Opus endpoint 400s on temperature ("does not support the temperature
    parameter") while its Sonnet and Haiku endpoints accept it. Asking and then dropping the
    parameter beats keeping a per-model capability table in sync with a workspace's endpoints -
    and a hard failure here would take out the critic, whose whole job is to rescue a weak run.
    """
    try:
        return client.chat.completions.create(model=model, messages=messages, **kwargs)
    except BadRequestError as exc:
        if "temperature" not in kwargs or "temperature" not in str(exc):
            raise
        print(f"  [llm] {model} rejects the temperature parameter; retrying without it")
        kwargs.pop("temperature")
        return client.chat.completions.create(model=model, messages=messages, **kwargs)


def _message_text(msg) -> str:
    """Flatten an assistant message's content to text.

    Most endpoints return a plain string, but Databricks' Claude Opus endpoint returns a list
    of content blocks. Normalizing here keeps that shape from leaking into _parse_json() and
    every caller downstream.
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


def _parse_json(raw: str) -> dict | None:
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
    return None


def chat_json(model: str, messages: list[dict]) -> dict | None:
    """One JSON-mode completion, retried once. Returns None if no attempt produced JSON.

    Empty completions do happen - a thinking model under GPU pressure can return nothing
    at all. Observed failures were transient (the same call replayed fine), so a retry
    usually clears them. The retry nudges temperature off 0 because temp-0 replays are
    byte-identical, which would just reproduce a deterministically-bad completion.

    response_format={"type": "json_object"} is Ollama-only: Databricks' Claude serving
    endpoints 400 on it (INVALID_PARAMETER_VALUE). Every prompt already asks for JSON-only
    output, and _parse_json() below already falls back to extracting the outermost {...}
    from a prose/fenced reply, so skipping the param for Databricks costs nothing.

    Callers must treat None as "no answer", never as a negative answer.
    """
    kwargs = {"response_format": {"type": "json_object"}} if config.LLM_BACKEND == "ollama" else {}
    for attempt in range(config.CHAT_JSON_ATTEMPTS):
        resp = _create(
            model,
            messages,
            temperature=0 if attempt == 0 else 0.3,
            **kwargs,
        )
        raw = _message_text(resp.choices[0].message)
        parsed = _parse_json(raw)
        if parsed is not None:
            return parsed
        print(
            f"  [llm] {model} returned non-JSON "
            f"(attempt {attempt + 1}/{config.CHAT_JSON_ATTEMPTS}): {raw[:120]!r}"
        )
    return None


def chat_tools(model: str, messages: list[dict], tools: list[dict]):
    """One tool-calling completion. Returns the assistant message."""
    resp = _create(model, messages, tools=tools, temperature=0)
    return resp.choices[0].message


def to_message_dict(msg) -> dict:
    """Serialize an assistant message for the next request.

    Drops Ollama's non-standard `reasoning` field so thinking models stay
    swappable without touching the agents.
    """
    out: dict = {"role": "assistant", "content": _message_text(msg)}
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
