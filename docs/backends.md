# Backends

Beegent talks to LLMs through a connector. Four ship with it: **Ollama** (the default, local, no
key), **Groq** and **Mistral** (hosted, each needing only an API key) and **Databricks** (a worked
example of a hosted one behind a workspace). Each connector owns its own models, credentials and
quirks.

## Ollama

```bash
cp .env.example .env
# set OLLAMA_BASE_URL if needed (it defaults to `http://localhost:11434`)

ollama pull deepseek-r1:14b    # planner, critic
ollama pull qwen3:8b           # geofetch
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Defaults: `deepseek-r1:14b` (planner, critic) · `qwen3:8b` (geofetch).

The context length matters: tool results are verbose, and on a small context Ollama silently drops
the *oldest* messages — the system prompt and the task — so the agent forgets its goal. It looks
like a model failure and is a context failure.

## Groq

```bash
cp .env.example .env
# set GROQ_API_KEY
# set GROQ_BASE_URL if needed (it defaults to `https://api.groq.com/openai/v1`)

uv run python -m beegent.run --backend groq \
  --country Kenya --use-case "administrative boundaries for a flood-response dashboard"
```

Defaults: `openai/gpt-oss-120b` (planner, geofetch, critic).

## Mistral

```bash
cp .env.example .env
# set MISTRAL_API_KEY
# set MISTRAL_BASE_URL if needed (it defaults to `https://api.mistral.ai/v1`)

uv run python -m beegent.run --backend mistral \
  --country Kenya --use-case "administrative boundaries for a flood-response dashboard"
```

Defaults: `mistral-small-latest` (planner, geofetch, critic).

## Databricks

```bash
cp .env.example .env
# set DATABRICKS_HOST and DATABRICKS_TOKEN

uv run python -m beegent.run --backend databricks \
  --country Kenya --use-case "administrative boundaries for a flood-response dashboard"
```

Defaults: `databricks-claude-sonnet-5` (planner) · `databricks-claude-sonnet-4-6` (geofetch) ·
`databricks-claude-opus-5` (critic).

These are **serving-endpoint names, not model names**. A 404 almost always means the endpoint is
not called that in your workspace — check `GET /api/2.0/serving-endpoints`.

## Choosing a backend for the web UI

`beegent-ui` takes no flags — a backend is one variable, inline or in `.env`:

```bash
LLM_BACKEND=groq       uv run beegent-ui
PLANNER_MODEL=qwen3:8b uv run beegent-ui   # still Ollama, but a lighter chat
```

The chat front door uses the **planner** role, so `PLANNER_MODEL` decides whether the first message
answers in seconds or minutes — on Ollama that role defaults to `deepseek-r1:14b`, a 14B reasoning
model doing a one-line classification, which is why a hosted backend feels so much quicker.
`validate()` runs before the server binds, so a bad key or endpoint fails at startup rather than
hanging the first message.

## Models need tuning

The defaults are a starting point, not a recommendation. Swap them per role and judge on
`totals.by_role` rather than on price per token.

Geofetch is where ~99% of a run's tokens go, and where a cheaper model can cost more. Same use
case, same portal, only the geofetch model changed:

| | `haiku-4-5` | `sonnet-4-6` |
|---|---|---|
| tokens | 608,987 | **304,792** |
| verified downloads | 0 | **2** |
| iterations | 2, still nothing | **1** |

Cheaper per call, twice as expensive per run — the weaker model spent its request budget guessing
path names instead of reading the service description it had already fetched. Nothing else differed.

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


### Token accounting

Token field names are a provider convention, not a standard. If your backend reports
`input_tokens`/`output_tokens` rather than `prompt_tokens`/`completion_tokens`, override
`_token_usage()` — it is the single place any token number is read, so one override covers both
chat methods.

## Search needs no backend

Web search is independent of the LLM backend and needs no key: a DuckDuckGo HTML → DuckDuckGo Lite
→ Bing fallback chain.

## Neither do embeddings

Catalog matching uses a local ONNX model through the optional `fastembed` extra, so it needs no
API, no key, and nothing from your backend — there is no `embedder` role. Install it with
`uv sync --extra embeddings`; without it, links are stored with no vector and the catalog simply
returns nothing.

`EMBED_MODEL` selects the model. Changing it invalidates every vector already stored — they are
excluded from matching rather than compared — so the next run re-embeds them for you, then warns
if `CATALOG_MIN_RELEVANCE` or `MEMORY_MIN_RELEVANCE` no longer fits the data under the new model,
naming the value it recommends. Both are environment variables, so adopting one needs no code
change: `CATALOG_MIN_RELEVANCE=0.71 uv run python -m beegent.run ...`.
