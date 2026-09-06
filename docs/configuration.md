# Configuration

Two kinds of setting, split by who owns them:

- **Pipeline behaviour** lives in `beegent/config.py` — how deep to search, how much to keep.
- **Models and credentials** live on the connector for each backend, so adding a backend never
  edits `config.py`. See [Backends](backends.md).

## Choosing backend and models

Every stage asks for a *role*: `planner`, `geofetch`, `critic`.

```bash
uv run python -m beegent.run --country France --use-case "..." \
  --backend ollama \
  --planner-model deepseek-r1:14b \
  --geofetch-model qwen3:8b \
  --critic-model deepseek-r1:14b
```

**Precedence, highest wins:**

| | Example |
|---|---|
| 1. CLI flag | `--geofetch-model qwen3:14b` |
| 2. Environment variable | `GEOFETCH_MODEL=qwen3:14b` |
| 3. `.env` file | `GEOFETCH_MODEL=qwen3:14b` |
| 4. The connector's defaults | `OllamaConnector.DEFAULT_MODELS` |

Overriding one role leaves the others alone. The run banner and `candidate_list.json` both record
what was actually used, so you never have to reconstruct it.

## Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--country` | required | The country to search for |
| `--use-case` | required | Free-text description of the data you need |
| `--out` | `candidate_list.json` | Where to write results |
| `--backend` | `ollama` | Which connector to use |
| `--planner-model` | connector default | Model for the planner |
| `--geofetch-model` | connector default | Model for the fetch agent |
| `--critic-model` | connector default | Model for the critic |

## Environment variable options

An alternative to the flags above, and **lower precedence** — a CLI flag always wins over an
environment variable. Set them dynamically for one run (`GEOFETCH_MODEL=qwen3:8b uv run ...`) or
put them in `.env` (see `.env.example`); real environment variables win over `.env`.

| Variable | Purpose |
|---|---|
| `LLM_BACKEND` | Which connector to use, if `--backend` is not given |
| `PLANNER_MODEL`, `GEOFETCH_MODEL`, `CRITIC_MODEL` | Per-role model override |
| `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | Backend credentials — see [Backends](backends.md) |
| Any integer setting in `beegent/config.py` | Overrides that pipeline tunable — see below |

## Pipeline settings

All in `beegent/config.py`. **Every integer setting below is overridable by an environment
variable of the same name**, with the same precedence as the model variables — so
`MAX_ANGLES=1 uv run python -m beegent.run ...` works without editing the file. A value that is
not a positive integer fails at startup rather than silently falling back.

`CATALOG_MIN_RELEVANCE` (`0.30`) is the cosine a stored dataset must reach to be offered — measured
with `tools/calibrate_relevance.py`, and re-measure it whenever `EMBED_MODEL` changes.

Changing `EMBED_MODEL` also invalidates every vector already stored: they are excluded from matching
rather than compared, so the catalog goes quiet until you run
`uv run python -m tools.backfill_embeddings`.

Three settings are deliberately not overridable: `PROBE_BYTES` (16 is a correctness floor — the
GeoPackage magic signature is exactly 16 bytes), and `CONFIDENCE_BY_REPORT` / `CONFIDENCE_DEFAULT`.

### Breadth and depth

| Setting | Default | Effect |
|---|---|---|
| `MAX_ANGLES` | `3` | Angles per iteration. **The main cost lever** — each angle is a full agent run |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Catalog matching model. Empty disables embeddings. Changing it invalidates every stored vector |
| `BEEGENT_DB` | `~/.beegent/memory.db` | Cross-run memory. On by default; `""` disables it. `~` is expanded, so it works from `.env` too |
| `CATALOG_FRESH_DAYS` | `7` | A stored link newer than this answers the run outright — no planner, no agent. `0` disables it |
| `MAX_CATALOG_PROBES` | `3` | Stored links re-probed per run before any angle starts |
| `MEMORY_RUNS` | `3` | Past runs for the same country replayed to the planner, when a store is configured |
| `MAX_ITERATIONS` | `2` | How many times the critic may send the run back to the planner |
| `MAX_FINAL_CANDIDATES` | `3` | Cap on the candidate list |

### The geofetch agent

| Setting | Default | Effect |
|---|---|---|
| `GEOFETCH_MAX_STEPS` | `20` | LLM calls per angle |
| `GEOFETCH_MAX_HTTP_REQUESTS` | `50` | Web requests per angle, across all tools |
| `GEOFETCH_MAX_REPEATS` | `3` | Identical calls tolerated before the angle is abandoned |
| `GEOFETCH_MIN_EFFORT_REQUESTS` | `5` | A give-up filed before this many requests is bounced once |

### Context and truncation

These control what enters the conversation, which is what drives token cost — see
[Performance](performance.md).

| Setting | Default | Effect |
|---|---|---|
| `TOOL_RESULT_MAX_CHARS` | `8000` | Hard cap on one tool result. **The real input-token lever** |
| `KEEP_FULL_TOOL_RESULTS` | `3` | Recent tool results kept at full length |
| `TRIM_TOOL_TO` | `500` | Older tool results are trimmed to this |
| `TRIM_ASSISTANT_TO` | `800` | Cap on a kept assistant message |
| `MAX_BODY_BYTES` | `20000` | Raw XML/JSON kept per page |
| `MAX_TEXT_CHARS` | `2000` | Extracted HTML text kept per page |
| `MAX_LINKS` | `80` | Links reported per page |
| `MAX_URLS_FOUND` | `60` | Entries in `urls_found` |
| `PROBE_TEXT_BYTES` | `65536` | Second read when a probed payload is JSON/XML, so a feature service can be parsed rather than guessed from its first byte |

### Network and retries

| Setting | Default | Effect |
|---|---|---|
| `HTTP_TIMEOUT` | `20` | Per web request, seconds |
| `LLM_TIMEOUT` | `180` | Per LLM call, seconds |
| `LLM_MAX_RETRIES` | `3` | SDK retries on 429/5xx/timeouts |
| `CHAT_JSON_ATTEMPTS` | `2` | Retries for a JSON-mode call that returns unparseable output |

Watch `LLM_TIMEOUT` and `LLM_MAX_RETRIES` together: the worst case for a single call is
`(LLM_MAX_RETRIES + 1) × LLM_TIMEOUT`, and angles run one after another.

### Scoring

| Setting | Default | Effect |
|---|---|---|
| `CONFIDENCE_BY_REPORT` | `{high: 0.95, medium: 0.8, low: 0.7}` | Maps the model's self-assessment |
| `CONFIDENCE_DEFAULT` | `0.7` | Used when the self-assessment is unrecognised |

## Development tools

Two scripts in `tools/`, neither part of the pipeline:

| Command | What it does |
|---|---|
| `python -m tools.calibrate_relevance ["use case" ...]` | Scores your stored links against each query, prints the band the data implies, and suggests `CATALOG_MIN_RELEVANCE`. Re-run it whenever `EMBED_MODEL` changes |
| `python -m tools.backfill_embeddings` | Embeds stored links with no vector, or one from a different model. Required after an `EMBED_MODEL` change, or the catalog goes quiet |

`calibrate_relevance` exits non-zero when no single threshold satisfies every query, printing the
conflicting bands rather than averaging them into a number that works for neither.
