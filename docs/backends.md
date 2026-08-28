# Backends and models

`beegent/connectors/` ships **Ollama** (the default, local, no key) and **Databricks** (the worked
example). Each connector owns its own models, credentials and quirks — `beegent/config.py` never
learns a backend's name.

## Ollama (default)

Runs against `http://localhost:11434`. Pull the models it expects:

```bash
ollama pull deepseek-r1:14b   # planner, critic
ollama pull qwen3:8b          # geofetch - the tool-calling agent
```

Give local models room. Tool results are verbose and Ollama's default context is small:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

On a small context Ollama silently drops the **oldest** messages — the system prompt and the task
itself — and the agent drifts off-goal in a way that looks like a model failure but is a context
failure.

`qwen3:8b` (~5.2GB) is deliberate for geofetch: it has the most reliable tool calling here, and
fits an 8GB card with room for a 16k context. Do **not** "upgrade" to `qwen3:14b` — it was tried
and measured at ~10GB resident, 39% spilled to CPU, and it blew past `LLM_TIMEOUT` on a single call.

## Databricks

```bash
cp .env.example .env
# edit .env: set DATABRICKS_HOST and DATABRICKS_TOKEN
LLM_BACKEND=databricks uv run python -m beegent.run --country Kenya \
  --use-case "administrative boundaries for a flood-response dashboard"
```

Defaults switch automatically to `databricks-claude-sonnet-4-6` / `databricks-claude-haiku-4-5` /
`databricks-claude-opus-5` (planner / geofetch / critic).

These are **serving-endpoint names, not bare model names** — a 404 usually means the endpoint is
not named that in your workspace. Check `GET /api/2.0/serving-endpoints`.

`connector.validate()` runs once at startup, so a missing token or a wrong endpoint name fails
*before* the run spends anything rather than as a 404 mid-angle.

## Choosing a backend and per-role models

Each pipeline stage asks for a **role** (`planner`, `geofetch`, `critic`) and the connector resolves
it to a model. Pick either from the command line:

```bash
uv run python -m beegent.run --country France --use-case "..." \
  --backend databricks --geofetch-model databricks-claude-sonnet-4-6
```

**Precedence, highest first:** CLI flag → environment variable → `.env` → the connector's
`DEFAULT_MODELS`. Overriding one role leaves the others on their defaults, and both the run banner
and `candidate_list.json` report what was *actually* used.

One `--<role>-model` flag is generated per role, so adding a role to `LLMConnector.ROLES` gives it
a CLI flag for free.

The geofetch step needs **native function calling** and is where model quality shows most — it is
the agent that has to reason its way from a landing page to a file.

## Adding a backend

A backend that speaks the OpenAI wire format is a subclass plus one registry entry:

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

CONNECTORS["mine"] = MyConnector     # now LLM_BACKEND=mine works
```

`validate()` is abstract, so every connector must answer it. A backend with genuinely nothing to
check says `return None` explicitly — there is no permissive default, because a connector that
silently inherited one would report itself validated while checking nothing.

A backend with a **different wire format** subclasses `LLMConnector` directly and implements
`chat_json()` and `chat_tools()` itself. Both return `(result, TokenUsage)`; mapping the response
into that shape is the whole integration job. `beegent/connectors/databricks.py` is the worked
example — read it to see what a real one costs.

Token field names are a provider convention, not a standard. A backend that spells them
`input_tokens`/`output_tokens` overrides `_token_usage()` and nothing else — it is the single place
any token number is read.

## Search needs no key at all

`WebTools.web_search()` is a keyless DuckDuckGo HTML → DuckDuckGo Lite → Bing fallback chain. A paid
provider was tried here and dropped: its ranking was not better (its top hit for a real query was
the file format's spec page), so it bought nothing.
