# Backends

Beegent talks to LLMs through a connector. Two ship with it: **Ollama** (the default, local, no
key) and **Databricks** (a worked example of a hosted one). Each connector owns its own models,
credentials and quirks.

## Ollama

The default. Runs against `http://localhost:11434`, override with `OLLAMA_BASE_URL`.

```bash
ollama pull deepseek-r1:14b    # planner, critic
ollama pull qwen3:8b           # geofetch
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

The context length matters: tool results are verbose, and on a small context Ollama silently drops
the *oldest* messages — the system prompt and the task — so the agent forgets its goal. It looks
like a model failure and is a context failure.

`qwen3:8b` is the geofetch default for a reason. It has the most reliable tool calling of the local
models tried here, and at ~5.2 GB it fits an 8 GB card alongside a 16k context. `qwen3:14b` was
measured at ~10 GB resident with 39% spilled to CPU, and exceeded `LLM_TIMEOUT` on a single call.

## Databricks

```bash
cp .env.example .env
# set DATABRICKS_HOST and DATABRICKS_TOKEN

uv run python -m beegent.run --backend databricks \
  --country Kenya --use-case "administrative boundaries for a flood-response dashboard"
```

Defaults are `databricks-claude-sonnet-4-6` (planner), `databricks-claude-haiku-4-5` (geofetch)
and `databricks-claude-opus-5` (critic).

These are **serving-endpoint names, not model names**. A 404 almost always means the endpoint is
not called that in your workspace — check `GET /api/2.0/serving-endpoints`.

Credentials are validated once at startup, so a bad token or a wrong endpoint name fails before the
run spends anything rather than midway through.

## Writing a connector

If your backend speaks the OpenAI wire format, it is a subclass and one registry entry:

```python
import os
from beegent.connectors import CONNECTORS, OpenAICompatConnector

class MyConnector(OpenAICompatConnector):
    provider = "mine"
    DEFAULT_MODELS = {"planner": "...", "geofetch": "...", "critic": "..."}

    def __init__(self, log=None):
        super().__init__(base_url="https://my-gateway/v1",
                         api_key=os.environ.get("MY_TOKEN", ""), log=log)

    def validate(self):
        return None if os.environ.get("MY_TOKEN") else "mine backend needs MY_TOKEN"

CONNECTORS["mine"] = MyConnector      # LLM_BACKEND=mine now works
```

Nothing else in Beegent changes. Its models, credentials and quirks all live in that one file, and
`beegent/config.py` never learns the backend exists.

### Required methods

| Method | Returns | Notes |
|---|---|---|
| `chat_json(model, messages)` | `(data, TokenUsage)` | `data` of `None` means "no answer", never a negative answer |
| `chat_tools(model, messages, tools)` | `(message, TokenUsage)` | Needs native function calling |
| `validate()` | `str \| None` | `None` when ready, otherwise the error to show the user |

`validate()` is abstract on purpose — there is no permissive default, because a connector that
inherited one would report itself validated while checking nothing. A backend with genuinely
nothing to check returns `None` explicitly.

Both chat methods return the same 2-tuple shape. **Unpack before testing the result** — a tuple is
always truthy, so `if chat_json(...)` would read a `None` answer as an answer.

### A different wire format

Subclass `LLMConnector` directly and implement `chat_json` and `chat_tools` yourself. Mapping the
response into `(result, TokenUsage)` is the whole integration job.

### Token accounting

Token field names are a provider convention, not a standard. If your backend reports
`input_tokens`/`output_tokens` rather than `prompt_tokens`/`completion_tokens`, override
`_token_usage()` — it is the single place any token number is read, so one override covers both
chat methods.

Do not guess field names for a provider you have not observed: a wrong guess produces a zero that
is indistinguishable from "not reported".

## Search needs no backend

Web search is independent of the LLM backend and needs no key: a DuckDuckGo HTML → DuckDuckGo Lite
→ Bing fallback chain. A paid search provider was tried and dropped because its ranking was not
better on real queries.
