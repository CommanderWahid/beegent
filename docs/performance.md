# Performance

Every number here was measured on a real run.

## What a run can cost

| Stage | Per unit | Worst case per run |
|---|---|---|
| Planner | 1 LLM call | 2 |
| Geofetch | ≤20 LLM calls, ≤50 HTTP requests **per angle** | 3 angles × 2 iterations = **120 LLM calls** |
| Critic | 1 LLM call | 1 |

Know this number before pointing Beegent at a paid endpoint. Real runs land far below the ceiling,
but do not budget on that.

A measured full run: **177,629 tokens** (139,161 in / 38,468 out) across 5 angles and 28 HTTP
requests. A single angle that resolves cleanly costs about **12,900 tokens** over 5 steps.

## Why cost behaves the way it does

The LLM is stateless, so **the whole conversation is re-sent on every step**. Cost is quadratic in
step count and dominated by input — roughly 5:1 input to output on a measured angle.

The consequence: **what enters the conversation matters more than how long it runs**. One large
tool result is charged again on every subsequent step.

## The levers, in order of effect

### 1. The start URL (up to 12×)

Where an angle begins decides what it costs, by more than any setting:

| Start point | Result |
|---|---|
| Bulk file server (`.../data/latest/`) | **verified** in 5 steps, 16,027 tokens |
| Portal landing page | failed after 20 steps, 197,883 tokens |

The planner is instructed to give the deepest concrete path it can name — a bulk directory, an
Atom/STAC/OGC/CSW endpoint, or an API base — and to avoid search pages, dataset landing pages and
publisher homepages. If your runs are expensive, look at the `url` values in the plan first.

### 2. `TOOL_RESULT_MAX_CHARS` (3.3× measured)

Not `GEOFETCH_MAX_STEPS`. One uncapped WFS GetCapabilities document measured 63,263 characters —
about 15,815 tokens — and was re-sent on every subsequent step. A single angle reached 861,963
input tokens and hit a rate limit.

Capping at 8,000 characters measured **3.3×** on a real path: 104,749 → 31,517 tokens.

### 3. `MAX_ANGLES`

Each angle is a full agent run. Halving the angles roughly halves the run.

### 4. Model choice

Geofetch dominates the bill — 172,771 of 177,629 tokens in the measured run. A cheaper model there
saves more than anywhere else, but it is also where quality matters most.

## Where the tokens went

```
cost:  5 angle(s), 28 request(s), 177,629 tokens (139,161 in / 38,468 out)
       geofetch 172,771  |  planner 3,884  |  critic 974
```

`run.totals["by_role"]` gives this split on every run. Two things it reveals:

- **The planner and critic are cheap in tokens but not in output.** A planner call is 893 in /
  1,049 out; a critic call 399 in / 505 out. Both are *output*-heavy, the inverse of geofetch,
  because reasoning models emit `<think>` blocks — and output is the expensive half on hosted
  endpoints.
- **A dead angle can cost as much as a successful one.** Dead ends carry their own `cost` block
  precisely so this is visible.

## Built-in protections

| Mechanism | What it prevents |
|---|---|
| Anti-loop abort | A stuck angle repeating one call. Previously observed burning ~180k tokens across 7 suppressions with no attempt to finish |
| History compaction | Old tool results growing the prompt forever |
| `urls_found` deduplication | Paying twice for URLs the `links` list already carries |
| Minimum effort | An agent giving up before trying the methodology, wasting the angle |

## Deliberately not done

**Angles run one at a time.** Parallelising them looks like free speedup and is not: Ollama serves
one request at a time by default, so calls queue anyway; `web_search` scrapes search engines with
no API key, and N simultaneous searches from one IP is the reliable way to get blocked; and rate
limits are limits on a *rate*, so compressing the same tokens into less time multiplies the
per-minute figure. Parallelism never reduces total tokens, only the window they land in.

**Prompt caching is deferred.** Only the system prompt, tool schemas and task are stable across
steps — about 1,539 tokens, charged on every call. Everything after them is rewritten by history
compaction, so it cannot be cached without giving up compaction, which is what keeps the prompt
from growing without bound in the first place.
