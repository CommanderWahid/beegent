# Build a proof-of-concept for the "discovery phase" of a geospatial data-scoping agent

## Context

We're standardizing how a geospatial data/AI team ingests new datasets. The full process is: business scoping -> technical scoping -> dev pipeline -> validation -> deployment. The bottleneck is the first part of technical scoping: for a given (country, use case), someone manually searches the internet for candidate data sources, opens a handful of endpoints, and judges which one is actually good enough to use.

This task is to build a **proof-of-concept for the complete discovery step** - all five parts of it - turning a `(country, use_case)` input into a short, deduplicated list of candidate data-source URLs worth investigating further, with a fallback path for when discovery comes up short. Nothing downstream of discovery (schema probing, scoring, column mapping, handoff docs) is in scope here.

**Goal: bare-minimum working code, not a polished system.** Prefer the simplest thing that proves the mechanism works end-to-end over completeness or architectural purity. It's fine to hardcode things, skip error handling in places, and leave TODOs for later. "Bare minimum" applies to *how* each piece is built, not to *which* pieces exist - all five steps below are in scope.

## Scope

**In scope for this POC - implement all five steps:**
1. **Planner** - takes the use case and produces 1 to n distinct search angles (n decided by the model itself, capped at 5 for cost control - not a fixed number).
2. **Catalog workers** - deterministic (no LLM) calls to a small hardcoded list of known open geospatial data catalogs.
3. **Search & explore worker** - one LLM-driven agent (per search angle) that does a web search and, only when a result looks like a homepage/landing page rather than an actual resource, fetches the page and follows one or two links deeper to find the real data page. Bounded tool-call budget (e.g. max 1 search + 3 page fetches per angle).
4. **Merge & triage classifier** - dedupes all candidates by URL/domain, drops obvious junk with a deterministic pre-filter, then runs a cheap LLM classification on what survives (real data endpoint vs. noise).
5. **Critic agent** - triggered only when triage's output looks weak (see "Escalation gate" below). Decides whether to send the planner back for another attempt (bounded, max 2 total escalations) or give up and flag the run for a human to search manually.

Final output is a JSON file either way: a normal run produces `candidate_list.json` with the surviving candidates; an escalated-and-still-empty run produces the same file shape plus a `status: "needs_human_review"` flag and a short reason.

**Explicitly out of scope for this POC** (do not build these yet):
- The periodic catalog-list maintenance job (health-check + scout agent that keeps the catalog list fresh over time). Unrelated to the per-request critic above.
- Everything downstream of discovery: schema probing, scoring/ranking against a rubric, column mapping, metadata/handoff packaging.
- Any UI or persistence layer beyond writing a JSON file. "Human review" here just means the JSON says so and the person reads it - no ticketing/notification system.
- Multi-vendor model orchestration - use one model provider (with two swappable backends, see "Models" below) for every LLM call in this POC. Adding other providers is future work.

## Architecture to implement

```
(country, use_case)
        |
        v
    Planner (LLM call)  <-------------------------------+
        |  -> produces 1..n search angles                 |  re-plan,
        v                                                   |  max 2 total attempts
   +---------------------------+                           |
   | Catalog workers (no LLM)  |   <- runs once             |
   +---------------------------+                           |
   +---------------------------+                           |
   | Search & explore worker   |   <- runs once per angle    |
   | (LLM + web_search + fetch)|                             |
   +---------------------------+                           |
        |                                                    |
        v                                                    |
    Merge & triage (dedupe + deterministic filter + LLM)      |
        |                                                    |
        v                                                    |
    Escalation gate (deterministic: enough signal?)           |
        |                          |                          |
       yes                        no                          |
        |                          v                          |
        |                    Critic agent (LLM call, rare) ---+
        |                          |
        |                     still not enough
        |                          v
        v                    status: needs_human_review
    candidate_list.json      candidate_list.json (+ status, reason)
```

The escalation gate is deterministic, not an LLM call - it's a simple threshold check (see "Implementation notes"). Only the critic itself is a model call, and it only runs when the gate fails.

## Data shapes

Keep these as simple dataclasses or TypedDicts - no need for a full framework.

```python
SearchAngle:
    description: str          # e.g. "national statistics office boundary data"
    channel_hint: str          # e.g. "catalog" | "web_search" | "national_geoportal"
    rationale: str

Candidate:
    url: str
    title: str
    source: str                # "catalog:hdx" | "search_explore" etc.
    hops: int                   # how many page-fetches it took to find this (0 = direct search hit)
    rationale: str
    confidence: float | None    # from the triage step

DiscoveryRun:
    country: str
    use_case: str
    iteration: int = 1           # bumps each time the critic sends it back to the planner
    max_iterations: int = 2      # hard cap - after this, stop re-planning and flag for human
    candidates: list[Candidate]
    status: str                  # "ok" | "needs_human_review"
    reason: str | None            # populated when status is needs_human_review
```

## Models

**Default backend: locally hosted models via Ollama** (already installed, running on `http://localhost:11434`). **Secondary backend (switchable): company-provided models served through Databricks' Unity AI Gateway.**

Both backends happen to speak the same wire format, which simplifies this a lot: Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`, and Databricks Model Serving/Unity AI Gateway also exposes an OpenAI-compatible API (`https://<workspace-host>/serving-endpoints/<endpoint-name>/invocations`, usable directly via the standard `openai` Python client or Databricks' `databricks-openai` package). So: use the `openai` Python SDK as the one client for every LLM call in this POC, and make the backend a config switch, not two different code paths.

```python
# one small factory, everything else just calls client.chat.completions.create(...)
def get_llm_client(backend: str = "ollama"):
    if backend == "ollama":
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # placeholder key, unused by Ollama
    elif backend == "databricks":
        return OpenAI(
            base_url=f"https://{DATABRICKS_HOST}/serving-endpoints",
            api_key=DATABRICKS_TOKEN,  # PAT for now; move to OAuth machine-to-machine token before anything production
        )
```

Don't build support for anything beyond these two backends in this POC - no direct Anthropic/OpenAI/Gemini API calls. The design (one client, swappable `base_url`/`api_key`) leaves room to add those later without a rewrite.

Tool use (web_search, fetch) for the search & explore worker needs to work through whichever local/Ollama model you pick - check that it actually supports tool calling reasonably well before committing to it; not all locally-hosted models are equally reliable at structured tool use, and this is worth a quick sanity check before writing the rest of the worker around it.

**Model picking is manual and fixed - never auto-detected or chosen at runtime.** No code that queries `ollama list` and picks "the best available one," no fallback-chain-of-models logic. Every model in play is an explicit, named value that's visible by reading one file or checking env - that's what makes a POC's results reproducible and comparable across runs.

Put all of it in one small `config.py`, as plain constants resolved from env vars with hardcoded defaults:

```python
# config.py
import os

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" | "databricks"

# Model name is backend-specific: an Ollama model tag when LLM_BACKEND=ollama,
# or a Databricks serving-endpoint name when LLM_BACKEND=databricks.
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "qwen3:32b")
SEARCH_EXPLORE_MODEL = os.environ.get("SEARCH_EXPLORE_MODEL", "qwen3:32b")
TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "qwen3:8b")     # small/cheap - runs once per candidate
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "qwen3:32b")     # rare calls, favor the strongest model you have pulled

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
```

Why this shape: the hardcoded default is the "fixed" part (change it by editing one line, and the change shows up in git history - useful when comparing which model handled a run well), the env var is the override for switching machines/backends without touching code (e.g. CI, or a teammate's setup later), and there's exactly one place to look to know what ran. Everything else in the codebase should import these constants rather than referencing model names or `os.environ` directly - that keeps model choice centralized instead of scattered across files.

If quick per-run experimentation turns out to matter (e.g. sweeping a few pulled Ollama models to see which handles tool-use best), layer optional argparse flags on top that override these same constants at invocation time (`--planner-model`, `--search-model`, etc.) - but skip that for the very first pass and add it only once you're actually doing that comparison.

Suggested model per role, but don't over-think this for a POC - reusing one capable model everywhere is fine to start:
- **Planner**: whichever available model is strongest at instruction-following/reasoning - low-volume, high-impact if it goes wrong.
- **Search & explore worker**: needs reliable tool use (web search + fetch) above all else; same model as planner is fine for v1 if it handles tool calls acceptably.
- **Merge & triage classifier**: runs once per surviving candidate, so volume is higher and the judgment call is simpler (real data endpoint vs. noise) - pick the smallest/fastest model that gives reliable yes/no answers. `TRIAGE_MODEL` is deliberately separate from the others for exactly this reason.
- **Critic agent**: runs rarely (only on escalation), and it's the highest-stakes call in the pipeline - it's the one deciding whether to keep trying or give up and hand the whole thing to a human. Don't cheap out on this one; if you have a noticeably stronger model pulled than what you're using for planner/search, use it here even if it's slower.

## Implementation notes / acceptable POC shortcuts

- **Catalog workers**: hardcode 2-3 catalogs to start (HDX/CKAN API, geoBoundaries API are the easiest to call with no auth). Don't build the full seed-catalog table yet.
- **Web search tool**: any working web search API/library is fine (e.g. an existing search API you have access to, or a simple HTTP call to a search provider). Don't build anything fancy here.
- **Fetch/browse tool**: a plain HTTP GET + basic HTML-to-text extraction (e.g. BeautifulSoup, strip tags) is enough. No headless browser, no JS rendering - note this as a known limitation, not something to solve now.
- **Triage/classification**: two layers, cheap first. A deterministic pre-filter (domain blocklist, obvious non-HTML/non-data content types) drops the easy junk for free before any model call. Whatever survives that goes to one `TRIAGE_MODEL` call per candidate - a short yes/no classification, not a full analysis.
- **Escalation gate**: deterministic, not a model call. Keep the threshold dead simple for v1 - e.g. fire the critic if fewer than 2 candidates survived triage, or if every surviving candidate came from a single worker/source. Tune the exact numbers later; the point for now is that the gate exists and is cheap to evaluate.
- **Critic agent**: one LLM call, given the original request, the search angles that were tried, and what triage produced (or didn't). It returns one of two decisions: `replan` (with a short note on what to try differently - e.g. "try the national statistics office angle instead of generic web search") or `needs_human_review` (with a short reason why). On `replan`, increment `DiscoveryRun.iteration`, call the planner again with that note added to its prompt, and re-run the pipeline. On reaching `max_iterations` (2) without success, stop and write `needs_human_review` regardless of what the critic says - the hard cap is not optional, it's what keeps a bad run from looping forever.
- **Parallelism**: sequential execution across search angles is fine for the POC. Don't spend time on async/concurrency unless it's trivial.
- **Persistence**: writing the final candidate list to a local JSON file is enough. No database.

## Suggested layout (lightweight - collapse into fewer files if that's faster)

```
discovery_poc/
  config.py            # backend + model constants
  planner.py           # generates search angles, accepts optional replan feedback
  catalog_workers.py   # deterministic catalog calls
  search_explore.py    # LLM + web_search + fetch loop
  merge_triage.py       # dedupe + deterministic filter + LLM classification
  critic.py             # escalation gate check + critic LLM call
  run.py                 # CLI: takes --country and --use-case, runs the full loop, writes candidate_list.json
```

A single `run.py` with everything inline is also acceptable for a true bare-minimum first pass - split into modules once it works end-to-end.

## How to validate it worked

Run it against one real example end-to-end, e.g.:

```
python run.py --country Kenya --use-case "administrative boundaries for a flood-response dashboard"
```

Success looks like: a `candidate_list.json` with somewhere between 1 and 6 deduplicated candidate URLs, each with a title and a short rationale, including at least one hit from the catalog workers and at least one hit from the search & explore worker, `status: "ok"`. Print a short summary to stdout as well (candidate count, top pick).

Then check the escalation path actually fires by running something deliberately thin, e.g. an obscure or made-up use case unlikely to have real sources:

```
python run.py --country Kenya --use-case "left-handed beekeeper density by micro-district"
```

Success here looks like either a `replan` happening once and then still coming up short, or an immediate `needs_human_review` - either way, `candidate_list.json` should end with `status: "needs_human_review"` and a `reason` string, and the run should terminate rather than loop indefinitely (confirm `iteration` never exceeds `max_iterations` in the output/logs).

Don't build retries, comprehensive logging, or tests beyond these two manual checks for v1 - the point is to prove the full mechanism (including the escalation path) works, not to productionize it yet.
