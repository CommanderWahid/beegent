![Scoutbee, an investigator bee mascot](assets/scoutbee_investigator.svg)

# Discovery-phase POC

Proof-of-concept for the **discovery step** of a geospatial data-scoping agent: turns a
`(country, use_case)` pair into a short, deduplicated list of candidate data-source URLs,
with an escalation path for when discovery comes up short. Spec:
[`discovery_phase_poc_prompt.md`](discovery_phase_poc_prompt.md).

Bare-minimum working code, not a polished system.

## Run it

```bash
uv sync
uv run python discovery_poc/run.py --country Kenya \
  --use-case "administrative boundaries for a flood-response dashboard"
```

Writes `candidate_list.json` and prints a summary. Requires Ollama running locally
(`http://localhost:11434`) with the models in `config.py` pulled.

## Pipeline

```
(country, use_case)
  -> planner            1..5 distinct search angles (count is the model's call)   [LLM]
  -> catalog workers    HDX CKAN + geoBoundaries, runs once                       [no LLM]
  -> search & explore   one agent per angle: 1 web search + up to 3 page fetches  [LLM + tools]
  -> merge & triage     dedupe -> MIME prefilter -> yes/no + publisher            [LLM]
  -> escalation gate    <2 survivors, or thin AND single-publisher                [no LLM]
  -> critic             replan (max 2 attempts) or needs_human_review             [LLM, rare]
```

The gate is a threshold check, not a model call. The `max_iterations` cap is unconditional:
on reaching it the run writes `needs_human_review` regardless of what the critic says.

**Diversity is measured on publisher**, which triage guesses per candidate alongside its
yes/no verdict — not on which worker found the candidate, and not on raw domain. Which
worker found something says nothing about quality (catalog workers legitimately come up
empty for use cases HDX doesn't cover), and raw domain would flag one national portal
hosting several agencies' datasets as undiverse. A single publisher only counts as failure
when the result is *also* thin (`SINGLE_PUBLISHER_MIN_CANDIDATES`): several distinct
high-confidence datasets from one good publisher is a fine outcome.

**The pre-filter keeps no lists.** It drops candidates whose MIME family (via stdlib
`mimetypes`) is image/video/audio or PDF — cases where a model call is obviously wasted —
and keeps everything else, including unknown types. There is no domain blocklist: judging
social media, forums, wikis and search-result pages is triage's job, named in its prompt
rather than matched against hosts nobody would maintain. Trade-off: office documents
(`.docx`, `.pptx`) and social URLs now cost one cheap triage call each instead of being free.

**Triage has three outcomes, not two.** Yes, no, and *no answer*. An LLM call that comes back
empty or unparseable is retried once (`CHAT_JSON_ATTEMPTS`); if it still produces no verdict,
the candidate is **excluded from `candidates`** — so junk can never ride in on a failed call —
and recorded in an `unresolved` array in the output, with `confidence: null` as the marker that
nothing judged it. Coercing a non-answer into a "no" silently deleted the best France candidate
once; coercing it into a "yes" would admit junk. Neither is acceptable, so it stays its own
outcome. If unresolved candidates outnumber survivors, the gate escalates rather than reporting
a confident `ok`.

**Re-planning adds to the candidate pool, it does not restart it.** Survivors from earlier
iterations are unioned into the next iteration's raw pool, win dedupe collisions, and skip
re-triage (they already passed). Unresolved candidates ride along too and *do* get re-triaged,
since they never got a verdict. A candidate that passed triage once never has to be re-found to
reach the final output.

| File | Role |
|---|---|
| `config.py` | backend, models, tunables — the only place any of them are named |
| `llm.py` | one `OpenAI` client, backend as a `base_url` switch (Ollama / Databricks) |
| `schemas.py` | `SearchAngle`, `Candidate`, `DiscoveryRun` |
| `planner.py` `catalog_workers.py` `search_explore.py` `merge_triage.py` `critic.py` | the five steps |
| `run.py` | CLI + orchestration loop |

## Models

Model choice is manual and fixed — edit `config.py` or set the env var; nothing is
auto-detected at runtime. **Bigger is not automatically better here.** Ablating the planner on
France/buildings: `llama3.1:8b` named IGN (the correct authoritative publisher) in 12 s with the
prompt unchanged, while `qwen3:14b` (84 s) and `deepseek-r1:14b` (53 s) both missed it — the
thinking models spend their reasoning abstracting into *categories* of publisher ("official
national open data portal", "academic institutions") and never name an institution. Prompted to
name one, `deepseek-r1:14b` invented a wrong name for IGN. Known limitation of the fix: models
do sometimes fabricate plausible-sounding agency names (e.g. "Kenya National Mapping Agency",
which is not a real body — Kenya's is the Survey of Kenya). Search is fuzzy enough that this
usually degrades gracefully, but the angle text is not a reliable source of institution names. Sizing note for an 8 GB RTX 4070 Laptop: `llama3.1:8b` fits
entirely in VRAM, tool-calls reliably (~8 s/turn) and classifies in ~3 s, while `qwen3:14b`
spills to CPU and always emits reasoning tokens (~90 s/call), which makes the per-angle
search loop crawl. The search & explore role needs a tool-capable model;
`deepseek-r1:14b` has no tool support, so it suits planner/critic only.

Switch backends with `LLM_BACKEND=databricks` plus `DATABRICKS_HOST` / `DATABRICKS_TOKEN`.

## Validated

- Kenya / administrative boundaries: 6 candidates, `status: "ok"`, both catalog and
  `search_explore` hits, ~110 s on `llama3.1:8b`.
- France / building vector data: 6 candidates across 4 publishers, `status: "ok"` on
  iteration 1 with **zero** catalog hits (HDX carries no French building footprints), top
  pick the official BDNB on data.gouv.fr. ~935 s on the thinking models. Two of the six sit
  on data.gouv.fr under different publishers (IGN, OpenStreetMap contributors) — the case
  domain-based diversity would have got wrong.
- Thin use case (`left-handed beekeeper density by micro-district`): gate fired at 1 survivor,
  critic returned `needs_human_review` with a reason, run terminated at iteration 1/2, ~130 s.
- Gate thresholds, re-plan branch, cross-iteration carry-over and the hard cap: covered by
  stubbed runs that make no LLM calls (publisher diversity vs. thinness, mirrors across
  domains, domain fallback for unknown publishers, and a survivor persisting through a
  forced `replan` into the final output).

## Known limitations

- `web_search` scrapes the DuckDuckGo Lite endpoint. Keyless and free, but unofficial —
  rate limits or markup changes will break it. Swap point is that one function.
- `fetch_page` is a plain GET with no JS rendering, so JS-only data portals look empty.
  Some sites also 403 the scraper or fail TLS verification; those fetches are logged and skipped.
- `COUNTRY_ISO3` in `catalog_workers.py` is a hardcoded ~12-country dict (TODO: `pycountry`).
- The Databricks backend is wired but untested — no host/token available here.
- No retries, no test suite, no logging framework: the two manual runs above are the check.

## Out of scope (per spec)

Catalog-list maintenance job; everything downstream of discovery (schema probing, scoring,
column mapping, handoff packaging); any UI or persistence beyond the JSON file; other model
providers.
