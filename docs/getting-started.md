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

## Run it in a container

Everything is built into the image — the Angular UI included — so this needs no checkout, no
Node, and no `uv` on the host:

```bash
docker compose up --build        # http://127.0.0.1:8000
```

Or without compose:

```bash
docker build -t beegent .
docker run --rm -p 127.0.0.1:8000:8000 -v beegent-data:/data   -e LLM_BACKEND=groq -e GROQ_API_KEY=... beegent
```

**`-p 127.0.0.1:8000:8000` is the access control.** There is no authentication in beegent, and
starting a run spends real model tokens. Publishing as `-p 8000:8000` binds every interface and
hands that to anyone who can reach the host.

Four things worth knowing before you run it:

- **One container, one process.** The run lock and the live log buffer live in memory in
  `api/runner.py`, so a second worker or replica would hold its own lock and answer for runs it
  never performed. No `--workers`, no scaling.
- **Ollama is not in the container.** `localhost` inside a container is the container, so the
  default backend has nothing to talk to. Use a hosted backend with a key, or point
  `OLLAMA_BASE_URL` at the host.
- **It exits on a bad backend, on purpose.** `validate()` runs before the server starts, so a
  missing key or a serving endpoint that does not exist stops the container rather than failing
  later mid-run. In Docker that looks like a restart loop; the message in `docker logs` is the
  real answer.
- **All state is on the `/data` volume** — `memory.db`, the preview cache, the embedding model
  cache, and `candidate_list.json`. Without the volume, every restart forgets what it verified.

The image carries `fastembed` and `onnxruntime` for catalog matching, which is most of its size.
Dropping the `embeddings` extra from the `pip install` line in the `Dockerfile` makes it
substantially smaller; `make_embedder()` returns `None` and the catalog simply stops matching by
meaning, rather than breaking.

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

## The second run is cheaper

The first run writes what it learned to `~/.beegent/memory.db` — the endpoints it verified, and the
critic's advice about what to try instead. Ask for the same thing again and the catalog answers
directly:

```
[catalog] 0.596 BAG building (Pand) footprints, whole country
[catalog] answered from memory - 1 link(s), no LLM call
```

That takes under a second and costs nothing, and the link is still independently re-probed before
it is handed back — a stored URL is never trusted on an old promise. After `CATALOG_FRESH_DAYS`
(7) the link is treated as a starting point again rather than an answer, because a live file is not
necessarily the current edition.

`BEEGENT_DB` moves the database; `BEEGENT_DB=""` turns the whole thing off.

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
