# Getting started

## Requirements

Python 3.10+, [uv](https://docs.astral.sh/uv/), and an LLM backend — the default is
[Ollama](https://ollama.com) locally, which needs no account and no key. Neither does search.

## Install

```bash
git clone https://github.com/CommanderWahid/beegent.git
cd beegent
uv sync                                    # the pipeline alone
uv sync --extra api --extra embeddings     # ...and the web UI

ollama pull deepseek-r1:14b                # planner and critic
ollama pull qwen3:8b                       # geofetch, the tool-calling agent
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

The context size matters: tool results are verbose, and on a small context Ollama silently drops
the *oldest* messages — the system prompt and the task — so the agent wanders off.

## Run it

The web UI is the usual way in.

```bash
cd ui && npm ci && npx ng build && cd ..
uv run beegent-ui                          # http://127.0.0.1:8000
```

It wants a use case and a real country before it spends anything; format and vintage are optional.
Localhost only, because a run costs real model calls. Three things it shows that
`candidate_list.json` cannot:

- **Logs** — the raw stream beside a live diagram of the pipeline, current step highlighted. A bee
  marks the three steps that call a model and cost tokens; a cog marks the deterministic ones.
- **Show on map** — draws any verified link a browser can render, answering what magic bytes
  cannot: whether the data covers the right country. Multi-layer files draw one layer at a time.
- **Catalog** — everything earlier runs verified, re-probed when you ask again.

### In a container

The image carries the built UI, so this needs no checkout, no Node and no `uv` on the host:

```bash
docker compose up --build        # http://127.0.0.1:8000
```

**`-p 127.0.0.1:8000:8000` is the access control** — there is no authentication, and a run spends
real tokens. Publishing as `-p 8000:8000` hands that to anyone who can reach the host.

- **One container, one process.** The run lock and the log buffer are in memory in
  `api/runner.py`; a second replica would hold its own lock and answer for runs it never performed.
- **Ollama is not in the container** — `localhost` there is the container. Use a hosted backend, or
  point `OLLAMA_BASE_URL` at the host.
- **It exits on a bad backend on purpose:** `validate()` runs before serving, so this looks like a
  restart loop. `docker logs` has the real answer.
- **All state is on the `/data` volume** — `memory.db`, the preview cache, `candidate_list.json`.

Most of the image size is `onnxruntime`, for catalog matching. Dropping the `embeddings` extra from
the `Dockerfile` shrinks it a lot; `make_embedder()` returns `None` and matching by meaning simply
stops, rather than breaking.

## Or from the command line

```bash
uv run python -m beegent.run \
  --country France \
  --use-case "building footprints as a geoparquet file"
```

A run takes a few minutes on local models. It prints progress as it goes and writes
`candidate_list.json` in the working directory. Use `--out` to write somewhere else.

## What you get

Each candidate carries a `resource_url` the harness downloaded 16 bytes of and identified by magic
number. `first_bytes_hex` starting `50415231` is the ASCII `PAR1` that begins every Parquet file —
the raw proof:

```json
{
  "url": "https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/",
  "resource_url": "https://cadastre.data.gouv.fr/.../france/cadastre.parquet",
  "confidence": 0.95,
  "verification": {
    "ok": true, "status": 206, "payload_type": "parquet",
    "first_bytes_hex": "50415231150415ba"
  }
}
```

Angles that found nothing land in `unresolved` with the reason they gave up, so a route that burned
budget is visible rather than silent. Every field is in [Output schema](output-schema.md).

## The second run is cheaper

The first run writes what it verified to `~/.beegent/memory.db`, along with the critic's advice.
Ask again and the catalog answers directly:

```
[catalog] 0.596 BAG building (Pand) footprints, whole country
[catalog] answered from memory - 1 link(s), no LLM call
```

Under a second, no model call — and the link is still re-probed before it is handed back, because a
stored URL is never trusted on an old promise. After `CATALOG_FRESH_DAYS` (7) it becomes a starting
point again rather than an answer: a live file is not necessarily the current edition.
`BEEGENT_DB=""` turns the whole thing off.

## Choosing models

Each stage asks for a *role* — `planner`, `geofetch`, `critic` — and the backend maps it to a
model. Override any of them:

```bash
uv run python -m beegent.run --country Kenya --use-case "admin boundaries"   --backend databricks --geofetch-model databricks-claude-sonnet-4-6
```

Geofetch is where model quality shows most — it has to reason from a landing page to a file, and
needs native function calling. See [Backends](backends.md).

## Next steps

- [Configuration](configuration.md) — the settings that control cost and depth
- [Performance](performance.md) — what a run costs before you point it at a paid endpoint
