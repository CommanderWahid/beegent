# Context management

The LLM is stateless, so **the whole conversation is re-sent on every step**. Cost is quadratic in
step count and dominated by input (~5:1 measured), and one large tool result is charged again on
every later step. So one question sits behind everything here: *what occupies the context window,
and what did it cost to put it there.*

Every number below was measured on a real run.

## What a run costs

| Stage | Per unit | Worst case per run |
|---|---|---|
| Planner | 1 LLM call | 2 |
| Geofetch | ≤20 LLM calls, ≤50 HTTP requests **per angle** | 3 angles × 2 iterations = **120 LLM calls** |
| Critic | 1 LLM call | `MAX_ITERATIONS - 1`, so 1 by default |

Know this before pointing Beegent at a paid endpoint. A measured full run: **177,629 tokens** over 5
angles. Real runs land far below the ceiling — but do not budget on that.

## Where the angle starts beats everything else

Measured on the same question: a bulk file server resolved in 5 steps for **16,027 tokens**; a portal
landing page failed after 20 steps and **197,883 tokens**. A **12× spread**, because a landing page
forces the agent to discover the download service first, and every page it reads on the way stays in
the window for the rest of the run.

So the planner names the deepest concrete path it can, and **debugging an expensive run starts with
the `url` values in the plan** — not with the tunables below.

## Keeping the window small

| Technique | Where | What it buys |
|---|---|---|
| **Tool-result capping** to `TOOL_RESULT_MAX_CHARS` (8,000), before it enters the conversation | `geofetch.py:_fit()` | **3.3× measured**, 104,749 → 31,517 tokens |
| **Sliding window** — last `KEEP_FULL_TOOL_RESULTS` (3) stay full, older trimmed to `TRIM_TOOL_TO` (500) | `geofetch.py:_compact_history()` | The prompt stops growing with step count |
| **Range-request probing** — 16 bytes (`PROBE_BYTES`) | `web_tools.py:probe_url()` | A multi-GB file verified without downloading it |
| **Streaming byte cap** at `MAX_BODY_BYTES` (20,000) | `web_tools.py:default_transport()` | A huge page is cut at the socket |
| **HTML distillation** — text + links only, capped at `MAX_TEXT_CHARS` (2,000) | `web_tools.py:summarize_html()` | The agent navigates by links, so prose is the cheapest thing to drop |

Two rules the code follows and you should too. **Trim structurally, not by cutting the tail** — the
most valuable field (`urls_found`) is serialised last, so a blind tail cut destroys it first. And
**put instructions before payloads** for the same reason.

## Reusing what earlier runs learned

The cheapest tokens are the ones an earlier run already paid for. Beegent writes what it learns to a
local SQLite store and retrieves it into the next run's prompts.

| Store | Question | Compares | Retrieved into |
|---|---|---|---|
| `links` | *already verified?* | use case × **dataset** | the candidate list, after a fresh re-probe |
| `runs` | *already failed?* | use case × **use case** | the planner as `feedback`; the critic as burned URLs + past advice |

Both match by cosine over locally-embedded vectors. Three things worth knowing:

- **A fresh catalog hit ends the run before the planner** — measured at **0 tokens in 0.7s**, against
  ~90,000 to rediscover. The link is still re-probed, so only the LLM calls are saved.
- **Run memory is narrowed by use case, not just country.** Netherlands held 3 runs across 2
  unrelated use cases, so country-only fed a boundaries run dead URLs from the footprints runs.
  Narrowed, it retrieves **1 of 3**.
- **Embeddings, not word overlap** — *"municipal limits and provincial borders"* and *"administrative
  boundaries (municipalities, provinces)"* share no token after stopwords: lexical **0.0**, cosine
  **0.597**. That is the case the catalog exists for.

**Retrieval costs no tokens.** Local ONNX embeddings, no API and no key; a dot product over a few
thousand rows is microseconds. The thresholds are measured, not guessed, and a run re-embeds and
re-checks them automatically when `EMBED_MODEL` changes — see
[Configuration](configuration.md#pipeline-settings).

## Bounding the run

| Technique | What it buys |
|---|---|
| **`MAX_ANGLES` (3)** | The main lever: each angle is a full agent run, so halving it roughly halves the run |
| **Per-angle caps** — `GEOFETCH_MAX_STEPS` (20), `GEOFETCH_MAX_HTTP_REQUESTS` (50) | Makes the ceiling above a ceiling, not an estimate |
| **One agent per angle** | A dead end's 20 steps never pollute the next angle's window |
| **Early exit** — `report_result` is the only way to finish | A clean angle ends at step 5, not step 20 |
| **Deterministic gate** before the critic runs | The common case answers itself for free |
| **Deterministic tools** — magic bytes, HTML, URL harvesting | Zero tokens for work a model would be paid to do |
| **Role-based model routing** | The cheap model runs the input-heavy worker; reasoning models stay in two one-shot calls |
| **Memoised repeats**, aborting at `GEOFETCH_MAX_REPEATS` (3) | One stuck run burned ~180k tokens with no attempt to finish |
| **Carry-forward across re-planning** | A verified candidate is never re-resolved |

## Reading a run's cost

```
cost:  5 angle(s), 28 request(s), 177,629 tokens (139,161 in / 38,468 out)
       geofetch 172,771  |  planner 3,884  |  critic 974
```

`run.totals["by_role"]` gives this split on every run. **A dead angle can cost as much as a
successful one** — dead ends carry their own `cost` block so this stays visible.

`cached_tokens` rides on the same buckets. Measured over 16 steps of real Groq traffic: **one hit,
of 1,024 tokens** — 10% of that run's input. It is a *subset* of `prompt_tokens`, never added to
`total_tokens`, and a `0` means "no hit reported", which is not the same as "no cache".

## Deliberately not done

**Angles run one at a time.** Ollama serves one request at a time so concurrency just queues;
keyless search scraping gets blocked from parallel requests; and a rate limit is a limit on a *rate*.
Parallelism shrinks the window it lands in, never the total.

**No model-written history summaries.** Compaction is deterministic truncation and costs nothing.
Summarising would spend tokens to save tokens, and put a paraphrase where the evidence was.

**No vector database.** Microseconds per query over a few thousand rows — the engine would buy
nothing `sqlite3` does not already give free.
