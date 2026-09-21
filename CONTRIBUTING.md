# Contributing

Contributions are very welcome — this is a working prototype, and most of what it still needs is
breadth rather than cleverness.

**You can work on almost all of it without spending a penny.** The test suite is offline by
design: no network, no API key, no model. The whole agent loop runs against a synthetic portal in
about two seconds.

## Set up

```bash
uv sync --extra api        # --extra api, or the web API's tests skip themselves
uv run pytest              # 376 cases, ~2s
uvx ruff check .
```

Both run in CI on Python 3.10, 3.11 and 3.12. That is the whole contract — if those pass, open the
pull request.

Before you commit, install the pre-commit hooks with `pre-commit install`. Make sure that the
pre-commit hooks pass on your local machine — they run the same ruff and pytest checks as CI, so a
clean run here means a clean run there.

## Things that would help

- **Try it on a country we have not.** You need no code for this: run it, and if the agent dead-ends
  it records *why* in `unresolved`. Paste that into an issue — a real dead end is more useful than a
  guess about one.
- **Add an LLM backend.** Subclass `OpenAICompatConnector` in `beegent/connectors/`, add one entry
  to the `CONNECTORS` registry, implement `validate()`. `databricks.py` is a worked example, and
  [docs/backends.md](docs/backends.md) walks through it.
- **Teach the map a new format.** Zipped shapefile and GeoParquet are recognised by the pipeline but
  not yet drawable in the browser.
- **Documentation.** If something here was unclear the first time, that is a bug in the docs.

Small pull requests are easier to say yes to than large ones. If you are planning something big,
open an issue first so we can agree the shape of it.

## Conventions

- **Comments and docstrings are one short line.** Rationale belongs in the docs.
- **Cross-module imports are absolute** — `from beegent import config`, never relative dots.
- **Pipeline tunables live in `beegent/config.py`.** Model names, URLs and credentials belong to the
  connector that owns them.
- **Nothing hardcodes a vendor, portal hostname or country.** Beegent has to work for any
  `(country, use_case)` on earth, and a portal table is always one country behind the next request.
  The agent tests run against a fictional country, so anything real that crept in would break them.

## The offline suite

A test that reaches the network is a bug, and this is enforced rather than hoped:

- an autouse fixture blocks `socket.connect`, so such a test fails loudly instead of passing on your
  machine and failing in CI;
- `WebTools` and `beegent/preview.py` both take an injectable `transport`;
- another autouse fixture blanks `config.BEEGENT_DB`, so no test touches your real store.

If your test needs a connector, stub its `validate()` — that method exists to ping a live endpoint.
`tests/` mirrors the package, one file per module. Per-test state is a fixture; the synthetic-portal
data is module-level and imported by name, because it is a constant rather than state.

## Before changing these

Not everything that looks arbitrary is. Two areas are load-bearing:

- **The verification guarantee** — every `resource_url` in the output was independently re-probed by
  the harness. `_handle_report()`'s guardrails are all covered by tests; read
  [docs/geofetch.md](docs/geofetch.md) first.
- **The similarity thresholds** are measured against a specific `EMBED_MODEL`, and a cosine is not a
  percentage. Change the embedding model and the run will tell you they no longer fit. See
  [docs/configuration.md](docs/configuration.md).

## The web UI

`api/` (FastAPI) and `ui/` (Angular) are **pure consumers**: they call `discover()`, read the store,
and render the result. The dependency runs one way — `beegent/` must never import them, and
`grep -rn "from api" beegent/` returning nothing is part of the check. The moment the UI
re-implements ranking or verification, the guarantee has two implementations and they will drift.

Run the halves separately so both reload:

```bash
LLM_BACKEND=groq uv run uvicorn api.main:app --reload --port 8000
cd ui && npx ng serve          # :4200, proxies /api across
```

(`uv run beegent-ui` does not reload — it serves the built UI and holds the modules it imported at
start, so an edit appears to do nothing until you restart it.)

The map preview has a handful of sharp edges — lazy-loading deck.gl, tile URL templates, the WASM
GeoPackage reader. Each is commented at the line it applies to in `ui/src/app/panes/link-map*.ts`;
read those before changing that file. Two API-level ones worth knowing anywhere: the API allows
**one run at a time**, and the log stream is an SSE tap on the existing `logging` calls — if you
need structured progress, `resolve_angle(angle, log=...)` already takes a callable.

## Questions

Open an issue. A question that turns out to be a documentation gap is a useful contribution too.
