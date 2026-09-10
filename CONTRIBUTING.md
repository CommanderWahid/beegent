# Contributing

```bash
uv sync
uv sync --extra api    # the web API's deps; without them its tests skip
uv run pytest          # 331 cases, ~2s
uvx ruff check .       # F (real errors) + ERA (commented-out code), configured in pyproject.toml
```

Both run in CI on Python 3.10, 3.11 and 3.12.

## The suite is offline by intent

No network, no API key, no LLM — so you can work on the agent loop without spending a token, and
**a test that reaches the network is a bug**. This is enforced, not merely intended:

- An autouse fixture blocks `socket.connect`, so a test that opens a connection fails with
  *"a test tried to reach the network"* rather than passing on your machine and failing in CI.
  (It cannot reach a subprocess, so a test that spawns one must stub its own dependencies.)
- `WebTools` takes an injectable `transport`. The whole agent loop is exercised against a synthetic
  portal defined in `tests/conftest.py`.
- Another autouse fixture blanks `config.BEEGENT_DB`, so no test touches your real
  `~/.beegent/memory.db` or loads an embedding model.

If your test needs a connector, stub its `validate()` — that method exists to ping a live endpoint.

`tests/` mirrors the package, one file per module under test. Per-test state is a **fixture**;
the synthetic-portal data is module-level and imported by name (`from tests.conftest import PORTAL`)
because it is a constant, not state.

## Conventions

- **Comments and docstrings are one short line.** Rationale belongs in the docs, not in the code.
- **Cross-module imports are absolute and fully qualified** — `from beegent import config`, never
  relative dots.
- **Nothing may hardcode a vendor, portal hostname or country.** Beegent has to work for any
  `(country, use_case)` on earth, and a portal table is always one country behind the next request.
  This is enforced by construction: the agent tests run against a fictional country, so anything
  real that crept into the harness would break them.
- **Pipeline tunables go in `beegent/config.py`**, not inline. Backend-specific things — model
  names, base URLs, credentials — belong to the connector that owns them.

## Adding an LLM backend

Subclass `OpenAICompatConnector` (or `LLMConnector` for a different wire format) in
`beegent/connectors/`, add one entry to the `CONNECTORS` registry, and implement `validate()` so a
bad credential fails before the run starts rather than mid-angle after tokens are spent. See
[docs/backends.md](docs/backends.md); `databricks.py` is the worked example.

## What not to change without measuring

Several values look arbitrary and are not — they carry findings from real runs, recorded in
[docs/configuration.md](docs/configuration.md) and [docs/performance.md](docs/performance.md).
The similarity thresholds in particular are **measured against a specific `EMBED_MODEL`**, and a
cosine is not a percentage. If you change the embedding model, the run will tell you the thresholds
no longer fit.

The verification guarantee is the point of the project: every `resource_url` in the output was
independently re-probed by the harness. `_handle_report()`'s guardrails are all load-bearing and all
covered by tests — see [docs/geofetch.md](docs/geofetch.md) before touching them.

## The web UI

`api/` (FastAPI) and `ui/` (Angular) are **pure consumers**: they call `discover()`, read
`SqliteStore`, and render `candidate_list.json`. The dependency runs one way only — `beegent/`
must never import them, and `grep -rn "from api" beegent/` returning nothing is part of the
check. The moment the UI re-implements ranking, confidence or verification, the guarantee has two
implementations and they will drift.

For development, run the two halves separately so both reload:

```bash
LLM_BACKEND=groq uv run uvicorn api.main:app --reload --port 8000
cd ui && npx ng serve          # :4200, proxies /api across
```

`uv run beegent-ui` does **not** reload: it holds the modules it imported at start, so an edit to
`api/` or `beegent/` does nothing until you restart it — a changed prompt or log line looks
missing when the process is simply older than the code. The trade with `--reload` is that
`main()` never runs, so you lose the startup `validate()` and the `[config]` line;
`GET /api/config` still answers.

Two things there that look like detail and are not. The API allows **one run at a time**, because
`discover()` interleaves store writes and Ollama serves one call at a time. And the log stream is
an SSE tap on the existing `logging` calls — do not invent a structured event protocol; if you
need one, `resolve_angle(angle, log=...)` already takes a callable.
