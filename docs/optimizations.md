# Cost and optimizations

Every number here was measured on a real run, not estimated.

## The cost ceiling

| | per unit | worst case per run |
|---|---|---|
| planner | 1 LLM call | 2 |
| geofetch | ≤20 LLM calls, ≤50 HTTP requests **per angle** | 3 angles × 2 iterations = **120 LLM calls** |
| critic | 1 LLM call | 1 |

Worth knowing before pointing this at a paid endpoint. `MAX_ANGLES` (`beegent/config.py`) is the
main lever. Real runs land far below the ceiling — but do not budget on that.

## Why cost behaves the way it does

The LLM is stateless, so **the whole conversation is re-sent on every step**. Cost is therefore
quadratic in step count, and dominated by input: a measured angle runs about **5:1 input to
output**, and a full run came in at 139,161 in / 38,468 out — input is ~78% of the tokens and more
than that of the bill.

The consequence that matters: *what enters the conversation is more expensive than how long the
conversation runs*. A single large tool result is charged again on every subsequent step.

## Planner — start-URL quality is the single biggest lever

The `url` an angle starts at decides what that angle costs, by more than any config value:

| start point | outcome |
|---|---|
| bulk file server (`.../data/latest/`) | **VERIFIED** in 5 steps, 16,027 tokens |
| portal landing page | failed at 20 steps, 197,883 tokens |

**12.3×.** Hence the `"url"` field guidance in `beegent/pipeline/planner.py`, which tells the
planner to give the deepest concrete path it can name — a bulk directory, an Atom/STAC/OGC/CSW
endpoint, or an API base — and to avoid search pages, dataset landing pages and publisher
homepages.

A planner call itself is cheap: **893 in / 1,049 out = 1,942 tokens**. Note it is *output*-heavy,
the inverse of geofetch, because the default planner is a reasoning model emitting `<think>`
blocks.

## Geofetch — `TOOL_RESULT_MAX_CHARS` is the input-token lever, not `GEOFETCH_MAX_STEPS`

Every kept tool result is re-sent on every step. One uncapped WFS GetCapabilities measured **63,263
characters ≈ 15,815 tokens** — charged *per step* for the rest of the run. A single angle reached
**861,963 input tokens** and hit a rate limit.

Capping at `TOOL_RESULT_MAX_CHARS = 8000` measured **3.3×** on a real path: 104,749 → 31,517 tokens.

Supporting the same goal:

- **`_compact_history()`** trims older tool results to `TRIM_TOOL_TO` (500) in place, keeping only
  the most recent `KEEP_FULL_TOOL_RESULTS` (3) at full length.
- **`MAX_BODY_BYTES`, `MAX_TEXT_CHARS`, `MAX_LINKS`, `MAX_URLS_FOUND`** cap what a page contributes
  before it ever becomes a tool result.

## Geofetch — abort a loop instead of paying for it

An observed stuck run hit **7 repeated-call suppressions and 0 attempts to finish**, burning
~180k tokens to produce nothing. Each further step re-sent the whole conversation, so the angle got
*more* expensive per step while making no progress.

`GEOFETCH_MAX_REPEATS = 3` now ends the angle with an honest failure. On a recent live run this
fired at step 13 of 20 on a dead angle that had already spent 38,163 tokens.

The guard applies to rescued plain-text tool calls too. It did not always: the rescue path used to
skip the bookkeeping entirely, so a model in template drift could repeat one call until the step
budget ran out — exactly the spend the guard exists to prevent.

## Geofetch — stop paying twice for the same URL

`fetch_page`'s `urls_found` now excludes URLs that the `links` list already carries. Before, an HTML
page paid for every anchor twice, on every step it stayed in history. What remains is the unique
value of the field: URLs in prose, XML metadata, or a JS bundle that no anchor would show.

## Critic — one gate condition

The escalation gate fires only when **nothing was verified at all**, so the expensive critic call is
skipped whenever a single file was probed successfully. Two broader conditions were removed for the
same reason: a thin-but-real result is a good outcome, not something worth second-guessing.

A critic call measures **399 in / 505 out = 904 tokens**, also output-heavy.

## Accounting — you can now see where it went

`run.totals["by_role"]` splits a run by stage. Planner and critic tokens used to be invisible
entirely, because `run.totals` summed only `Candidate.cost` and neither stage produces a candidate:

```
cost:  5 angle(s), 28 request(s), 177,629 tokens (139,161 in / 38,468 out)
       geofetch 172,771  |  planner 3,884  |  critic 974
```

Each call is counted exactly once, by the layer that can attribute it: `chat_tools()` usage is
summed per angle into `Candidate.cost`, `chat_json()` usage goes to a per-role meter in
`beegent/llm.py`. Metering both in both places would double-count every angle.

Planner and critic are ~2.7% of total tokens here but ~6.6% of *output* tokens, which is the
expensive half on hosted endpoints.

## Deliberately not done

**Angles run serially — do not parallelize them.** A `ThreadPoolExecutor` over the angle loop looks
like free speedup and is not. Ollama has one GPU and `OLLAMA_NUM_PARALLEL` defaults to 1, so calls
queue anyway; `web_search` scrapes DuckDuckGo/Bing with no API key against live bot detection, and N
searches from one IP inside a second is the reliable way to get the whole chain blocked; and rate
limits are limits on a *rate*, so compressing the same tokens into 1/N the time multiplies the
per-minute figure by N. Parallelism never reduces total requests or tokens, only the window they
land in.

**Prompt caching is deferred, not rejected.** Only messages 0–1 are ever cacheable — the system
prompt, tool schemas and task, about **1,539 tokens**, charged on all 120 calls of a worst-case run
(~182k tokens). That is the whole prize. Everything after the first trimmed message is a guaranteed
cache miss because `_compact_history()` rewrites already-sent messages in place, and weakening
compaction to fix that would reintroduce exactly the quadratic growth the caps above were tuned to
fight. Two facts still need live credentials to confirm: whether the gateway forwards cache markers,
and whether the stable prefix clears the minimum cacheable size.
