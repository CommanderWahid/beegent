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

**A repeated request costs nothing.** When the catalog holds a link verified within
`CATALOG_FRESH_DAYS`, the run ends before the planner is called: measured at **0 tokens in 0.7s**,
against ~90,000 for the same request discovered from scratch. The link is still re-probed, so the
saving is in LLM calls, not in verification.

A measured full run: **177,629 tokens** (139,161 in / 38,468 out) across 5 angles and 28 HTTP
requests. A single angle that resolves cleanly costs about **12,900 tokens** over 5 steps.

## Why cost behaves the way it does

The LLM is stateless, so **the whole conversation is re-sent on every step**. Cost is quadratic in
step count and dominated by input — roughly 5:1 input to output on a measured angle.

The consequence: **what enters the conversation matters more than how long it runs**. One large
tool result is charged again on every subsequent step.

## Optimisations

Before the code gets a say: **where an angle starts decides more than any of it.** <br>
The starting URL an angle is given matters more than every code-level optimisation combined. Measured: a bulk file server (.../data/latest/) resolved in 5 steps for 16,027 tokens, while a portal landing page failed after 20 steps and 197,883 tokens — a 12× spread on the same question. So the planner is instructed to name the deepest concrete path it can (bulk directory, Atom/STAC/OGC/CSW endpoint, API base) and avoid search pages, dataset landing pages and publisher homepages. Debugging an expensive run starts with the url values in the plan.

Everything below is applied by the code, ordered by how much it moves the bill.

### 1. Context-window management

The conversation is re-sent on every step, so this is the dominant category.

| Technique | Where | What it buys |
|---|---|---|
| **Tool-result capping** — one result is truncated to `TOOL_RESULT_MAX_CHARS` (8,000) *before* it enters the conversation | `geofetch.py:_fit()` | **3.3× measured**, 104,749 → 31,517 tokens. An uncapped WFS GetCapabilities measured 63,263 characters — ~15,815 tokens re-sent every step, and one angle reached 861,963 input tokens and a rate limit |
| **Sliding window over history** — the last `KEEP_FULL_TOOL_RESULTS` (3) tool results stay full, older ones are rewritten in place to `TRIM_TOOL_TO` (500) | `geofetch.py:_compact_history()` | The prompt stops growing with step count |
| **Assistant-message capping** at `TRIM_ASSISTANT_TO` (800) | same | Bounds a reasoning model's rambling in the replayed history |

### 2. Payload reduction at the source

Cheaper than trimming later, because what never arrives never costs a token.

| Technique | Where | What it buys |
|---|---|---|
| **Range-request probing** — `Range: bytes=0-15` (`PROBE_BYTES`) | `web_tools.py:probe_url()` | A multi-GB file is verified from its first 16 bytes and never downloaded |
| **Streaming byte cap** — the transport stops reading at `MAX_BODY_BYTES` (20,000) mid-response | `web_tools.py:default_transport()` | A huge page is cut off at the socket, not after it is in memory |
| **HTML distillation** — markup reduced to text + links, `<script>`/`<style>`/`<svg>` dropped, text capped at `MAX_TEXT_CHARS` (2,000) | `web_tools.py:summarize_html()` | The agent navigates by links, so prose is the cheapest thing to throw away |
| **Field caps** — `MAX_LINKS` (80), `MAX_URLS_FOUND` (60) | `web_tools.py:fetch_page()` | Bounds one page's contribution regardless of how link-heavy it is |
| **`urls_found` deduplication** against `links` | `web_tools.py:fetch_page()` | A URL already carried by the anchor list is not paid for twice |

### 3. Repeat suppression and reuse

| Technique | Where | What it buys |
|---|---|---|
| **Memoised identical calls** — an exactly repeated tool call is answered from memory with a "do something different" nudge | `geofetch.py:_run_call()`, `_call_seen` | No fetch, no HTTP budget spent on a result that cannot have changed |
| **Non-convergence abort** at `GEOFETCH_MAX_REPEATS` (3) | same | Ends a stuck angle. One observed run burned ~180k tokens across 7 suppressions with no attempt to finish |
| **Carry-forward across re-planning** — candidates verified in iteration N are never re-resolved in N+1 | `run.py:discover()` | Saves a full agent run per carried candidate |
| **Result dedupe** on `normalize_url(resource_url or url)` | `run.py:_rank()` | Two angles that land on the same file collapse to one candidate |

### 4. Budgets and early exit

| Technique | Where | What it buys |
|---|---|---|
| **Run-level caps** — `MAX_ANGLES` (3), `MAX_ITERATIONS` (2), `MAX_FINAL_CANDIDATES` (3) | `config.py` | `MAX_ANGLES` is the main lever: each angle is a full agent run, so halving it roughly halves the run |
| **Per-angle hard caps** — `GEOFETCH_MAX_STEPS` (20), `GEOFETCH_MAX_HTTP_REQUESTS` (50) | `geofetch.py`, `web_tools.py:_guard()` | The ceiling in the table above is a ceiling, not an estimate |
| **Early exit on success** — `report_result` is the only way to finish | `geofetch.py:_handle_report()` | A clean angle ends at step 5, not at step 20 |
| **Deterministic gate before an expensive call** — the critic runs only when nothing was verified at all | `critic.py:needs_escalation()` | The common case answers itself for free |
| **Deterministic tools instead of LLM parsing** — magic bytes, HTML, URL harvesting | `web_tools.py` | Zero tokens for work a model would otherwise be paid to do |
| **Role-based model routing** | connector `DEFAULT_MODELS` | The cheap model runs the input-heavy worker; reasoning models are confined to two one-shot calls |
| **Minimum-effort floor** — `GEOFETCH_MIN_EFFORT_REQUESTS` (5) | `geofetch.py:_handle_report()` | The one entry here that *spends*: it stops a whole angle's setup cost being thrown away on a premature give-up |

## Where the tokens went

```
cost:  5 angle(s), 28 request(s), 177,629 tokens (139,161 in / 38,468 out)
       geofetch 172,771  |  planner 3,884  |  critic 974
```

`run.totals["by_role"]` gives this split on every run. Three things it reveals:

- **The planner and critic are cheap in tokens but not in output.** 
- **A dead angle can cost as much as a successful one.** Dead ends carry their own `cost` block
  precisely so this is visible.
- **`cached_tokens` says how much of the input you were not charged full price for.** It rides on
  the same buckets, and the summary adds a line when it is non-zero:

  ```
  cached: 1,024 of the input tokens (10%)
  ```

## Deliberately not done

**Angles run one at a time** — parallelising them is not free speedup. Ollama serves one request at a time, so concurrent calls just queue; web_search scrapes engines keylessly, and N simultaneous searches from one IP is the reliable way to get blocked; and rate limits are limits on a rate, so the same tokens in less time multiplies the per-minute figure. Parallelism shrinks the window, never the total.

**Prompt caching is deferred** — only the system prompt, tool schemas and task (~1,539 tokens) are stable across steps and charged every call. Everything after them is rewritten by history compaction, so caching it would mean giving up the thing that keeps the prompt from growing without bound.
