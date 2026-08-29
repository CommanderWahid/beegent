# Getting started

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- An LLM backend. The default is [Ollama](https://ollama.com) running locally, which needs no
  account and no API key.

Web search needs no key either — Beegent scrapes a DuckDuckGo → DDG-lite → Bing fallback chain.

## Install

```bash
git clone https://github.com/CommanderWahid/beegent.git
cd beegent
uv sync
```

Pull the two default models and start Ollama with a larger context window:

```bash
ollama pull deepseek-r1:14b    # planner and critic
ollama pull qwen3:8b           # geofetch, the tool-calling agent

OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

The context size matters. Tool results are verbose, and on a small context Ollama silently drops
the *oldest* messages — which are the system prompt and the task — so the agent wanders off.

## Your first run

```bash
uv run python -m beegent.run \
  --country France \
  --use-case "building footprints as a geoparquet file"
```

A run takes a few minutes on local models. It prints progress as it goes and writes
`candidate_list.json` in the working directory. Use `--out` to write somewhere else.

## What you get

Each candidate carries a `resource_url` that the harness downloaded 16 bytes of and identified by
magic number. `first_bytes_hex` starting `50415231` is the ASCII `PAR1` that begins every Parquet
file — the raw proof:

```json
{
  "url": "https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/",
  "title": "Etalab cadastre bulk GeoParquet",
  "source": "geofetch",
  "confidence": 0.95,
  "resource_url": "https://cadastre.data.gouv.fr/.../france/cadastre.parquet",
  "verification": {
    "ok": true, "status": 206, "payload_type": "parquet",
    "first_bytes_hex": "50415231150415ba"
  }
}
```

Angles that found nothing are recorded under `unresolved` with the reason they gave up, so a route
that burned budget is visible rather than silent.

See [Output schema](output-schema.md) for every field.

## Choosing models

Each stage asks for a *role* — `planner`, `geofetch`, `critic` — and the backend maps it to a
model. Override any of them:

```bash
uv run python -m beegent.run --country Kenya --use-case "admin boundaries" \
  --backend databricks --geofetch-model databricks-claude-sonnet-4-6
```

Geofetch is where model quality shows most: it is the agent that has to reason from a landing page
to a file, and it needs native function calling. See [Backends](backends.md).

## Next steps

- [Configuration](configuration.md) — the settings that control cost and depth
- [Performance](performance.md) — what a run costs before you point it at a paid endpoint
