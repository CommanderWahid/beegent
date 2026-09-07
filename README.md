<div align="center">

# <img src="assets/beegent_logo.svg" alt="" height="72" valign="middle" /> Beegent

**Find geospatial data for any country and use case — and verify it actually downloads.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

[Getting started](docs/getting-started.md) ·
[Documentation](docs/README.md)

</div>

---

Give Beegent a country and a use case. It plans several routes to the data, sends an agent down
each one, and returns the download URLs it could **prove** are real.

```bash
uv run python -m beegent.run --country France --use-case "building footprints as a geoparquet file"
```

Every URL in the output has been independently re-fetched by the harness — right HTTP status, and
magic bytes matching the format that was asked for. A URL the model invented, or one that only
*looks* like a download, cannot get through.

## What a run looks like

One angle resolving against a live French government portal, from the start URL to a verified
file in five steps:

```
    [geofetch] task: url      = https://cadastre.data.gouv.fr/data/etalab-cadastre/latest/
    [geofetch]       dataset  = 'cadastral building footprints (batiments), whole France'
    [geofetch]       format   = 'GeoParquet'
    [geofetch]       vintage  = 'latest'
    [step 1] -> fetch_page({"url": ".../etalab-cadastre/latest/"})
    [step 2] -> fetch_page({"url": ".../etalab-cadastre/2026-06-01/geoparquet/"})
    [step 3] -> fetch_page({"url": ".../2026-06-01/geoparquet/france/"})
    [step 4] -> probe_url({"url": ".../geoparquet/france/cadastre.parquet"})
    [step 5] -> report_result({"found": true, "confidence": "high", ...})
    [geofetch] VERIFIED in 5 step(s), 5 request(s), 12,916 tokens
    [geofetch] path: fetch_page -> fetch_page -> fetch_page -> probe_url -> report_result
```

The agent was given a starting URL and a description. It found the current edition, walked to the
national file, and probed it. The harness then re-probed that URL itself before accepting it:

```json
{
  "resource_url": "https://cadastre.data.gouv.fr/.../france/cadastre.parquet",
  "confidence": 0.95,
  "verification": {
    "ok": true, "status": 206, "payload_type": "parquet",
    "first_bytes_hex": "50415231150415ba"
  }
}
```

`50415231` is `PAR1` — the four bytes that begin every Parquet file. That is the proof, read off
the wire.

## Why

Technical scoping is manual work: search for a country's data, open a dozen endpoints, work out
which are real, compare editions, write it up.

```
business scoping -> [technical scoping: BEEGENT] -> devs -> validation -> deployment
```

Beegent turns that into a review-ready draft. It does not replace the review — it does the
searching, and refuses to hand you anything it could not download.

## How it works

Four stages. A planner turns your use case into distinct **angles** — each one a concrete fetch
order, not a search query. An agent chases each angle with three tools (`fetch_page`,
`web_search`, `probe_url`) and can only finish by reporting a result, which the harness verifies
itself. If nothing was verified, a critic decides whether to re-plan or ask for a human.

```
planner -> catalogs -> geofetch (per angle) -> gate -> critic -> (replan | needs_human_review)
```

Nothing in the codebase knows about any specific portal, vendor or country.

See [Architecture](docs/architecture.md) and [The geofetch agent](docs/geofetch.md).

## Install

```bash
git clone https://github.com/CommanderWahid/beegent.git
cd beegent
uv sync

ollama pull deepseek-r1:14b qwen3:8b     # default backend is local Ollama
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

No API keys. Search is a keyless DuckDuckGo → Bing chain, and the default LLM backend runs
locally. <br>
Runs remember each other: a dataset found once is re-probed and returned in under a second, for
zero tokens ([how](docs/architecture.md#the-store)). <br>
You can add your backend (connector) or use the provided ones: Groq, Mistral, Databricks: [Backends](docs/backends.md).

Full walkthrough: [Getting started](docs/getting-started.md).

## Status

**Working prototype.** It resolves real datasets on live portals, and the verification guarantee
holds. Known limits, stated plainly:

- **Cross-run memory writes to `~/.beegent/memory.db`** from the first run. It is what lets a
  repeat request be answered from the catalog, and `BEEGENT_DB=""` disables it.
- **No JavaScript execution.** The agent routes *around* a JS app shell rather than through it,
  looking for the machine-readable service behind it ([how](docs/geofetch.md)). What stays out of
  reach is a download URL built by a client-side interaction; that is reported as an honest failure
  rather than guessed at.
- **Results depend on the model.** Small local models miss things a larger one finds.
- Portal APIs change. An angle that worked last month may dead-end today.

Pre-1.0: interfaces may change.

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first run, reading the result |
| [Configuration](docs/configuration.md) | Every setting, CLI flags, precedence |
| [Backends](docs/backends.md) | Ollama, Groq, Mistral, Databricks, writing a connector |
| [Architecture](docs/architecture.md) | The seven steps, the store, and how a run flows |
| [The geofetch agent](docs/geofetch.md) | The agent loop, its tools, the seven guardrails |
| [Output schema](docs/output-schema.md) | `candidate_list.json`, field by field |
| [Performance](docs/performance.md) | What a run costs and which settings move it |

## Development

The test suite is fully offline — no network, no API key, no LLM — so you can work on the agent
loop without spending a token:

```bash
uv run pytest                                       # 285 cases, ~1s
uvx ruff check --select F,ERA .
```

`tools/` holds one development utility, not part of the pipeline:

| | |
|---|---|
| `python -m tools.calibrate_relevance ["use case" ...]` | measures the similarity thresholds against the links you have stored — re-run it whenever `EMBED_MODEL` changes |

Runs on Python 3.10, 3.11 and 3.12.

## License

[Apache 2.0](LICENSE).
