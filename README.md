<div align="center">

# <img src="assets/beegent_logo.svg" alt="" height="72" valign="middle" /> Beegent

**Find geospatial data for any country and use case — and verify it actually downloads.**

[![CI](https://github.com/CommanderWahid/beegent/actions/workflows/ci.yml/badge.svg)](https://github.com/CommanderWahid/beegent/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

[Getting started](docs/getting-started.md) ·
[Documentation](docs/README.md)

<p align="center">
  <img src="assets/beegent_demo.webp" width="820"
       alt="A Beegent run: the prompt, the agent working, verified results, and the data drawn on a map" />
</p>

<sub>A real run, sped up — prompt, agent, verified downloads, and the data on a map.<br/>
Discovery takes a few minutes; a repeat request is answered from memory in under a second.</sub>

</div>

---

Tell Beegent a country and what you need the data for. It plans several routes to the data, sends
an agent down each one, and returns the download URLs it could **prove** are real.

Every URL it returns has been independently re-fetched by the harness — right HTTP status, and
magic bytes matching the format that was asked for. A URL the model invented, or one that only
*looks* like a download, cannot get through.

## Run it

```bash
uv sync --extra api --extra embeddings
cd ui && npm ci && npx ng build && cd ..
uv run beegent-ui                      # http://127.0.0.1:8000
```

You get an agent that asks what you need, a catalog of everything already verified, and the run's
log streamed live while it works. It wants a **use case and a real country** before it will spend
anything; format and vintage are optional.

Any verified link it can draw gets a **Show on map** button — which answers the one thing sixteen
magic bytes cannot: whether the data actually covers the right country. Multi-layer files list
their layers so you can look at one at a time.

Localhost only, and a run costs real model calls. `beegent-ui` has no flags — it reads its backend
and models from the environment or `.env` (`LLM_BACKEND`, `PLANNER_MODEL`, …) and shows the
resolved names at startup. See [Backends](docs/backends.md).

Prefer a container? `docker compose up --build` builds the UI and serves it on
`127.0.0.1:8000` with no toolchain on the host — see
[Getting started](docs/getting-started.md#run-it-in-a-container).

### Or from the command line

The same pipeline, writing `candidate_list.json` instead of drawing it:

```bash
uv run python -m beegent.run --country France --use-case "building footprints as a geoparquet file"
```

Flags for the backend, per-role models, angle count and output path are in
[Configuration](docs/configuration.md); the result format is in
[Output schema](docs/output-schema.md).

## Why

Technical scoping is manual work: search for a country's data, open a dozen endpoints, work out
which are real, compare editions, write it up.

```
business scoping -> [technical scoping: BEEGENT] -> devs -> validation -> deployment
```

Beegent turns that into a review-ready draft. It does not replace the review — it does the
searching, and refuses to hand you anything it could not download.

## How it works

A planner turns your use case into distinct **angles** — each one a concrete fetch order, not a
search query. An agent chases each angle with three tools (`fetch_page`, `web_search`,
`probe_url`) and can only finish by reporting a result, which the harness then verifies itself. If
nothing was verified, a critic decides whether to re-plan or ask for a human.

```
planner -> catalogs -> geofetch (per angle) -> gate -> critic -> (replan | needs_human_review)
```

Every verified link is remembered, so asking again for the same thing is matched by meaning and
re-probed rather than rediscovered — measured at **0 tokens in 0.7s**, against roughly 90,000
tokens to find it from scratch. Verification still happens every time; only the model calls are
saved.

Nothing in the codebase knows about any specific portal, vendor or country, and **no API key is
needed** — search is a keyless DuckDuckGo → Bing chain, and the default LLM backend runs locally.

## Documentation

- [Getting started](docs/getting-started.md) — install, first run, reading the result
- [Configuration](docs/configuration.md) — every setting, CLI flags, precedence
- [Backends](docs/backends.md) — Ollama, Groq, Mistral, Databricks, writing a connector
- [Architecture](docs/architecture.md) — the stages, the store, and how a run flows
- [The geofetch agent](docs/geofetch.md) — the agent loop, its tools, the seven guardrails
- [Output schema](docs/output-schema.md) — `candidate_list.json`, field by field
- [Context management](docs/performance.md) — what a run costs and what memory saves
- [CONTRIBUTING](CONTRIBUTING.md) — running the two halves in dev, and the test suite

## Status

**Working prototype.** It resolves real datasets on live portals, and the verification guarantee
holds. Known limits, stated plainly:

- **It writes to `~/.beegent/memory.db`** from the first run. That is what lets a repeat request be
  answered from the catalog; `BEEGENT_DB=""` disables it.
- **No JavaScript execution.** The agent routes *around* a JS app shell rather than through it,
  looking for the machine-readable service behind it ([how](docs/geofetch.md)). A download URL that
  only exists after a client-side interaction stays out of reach, and is reported as an honest
  failure rather than guessed at.
- **Results depend on the model.** Small local models miss things a larger one finds.
- Portal APIs change. An angle that worked last month may dead-end today.

Pre-1.0: interfaces may change.

## License

[Apache 2.0](LICENSE).
