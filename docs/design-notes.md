# Design notes

Why Beegent is built the way it is. These are the decisions that look wrong until explained, and
several were reversed once already. Read this before changing the agent loop or the guardrails.

For how to *use* the system, see the other pages in [docs/](README.md).

## Verification is the whole point

Anyone can ask an LLM for a download URL. The hard part is that models confidently produce URLs
that do not exist, and a plausible URL is worse than no URL — it looks like an answer.

So the model can only finish by calling `report_result`, and the harness re-probes the URL itself
with deterministic code. A fabricated URL cannot reach the output. The worst case is an honest
failure, which is a much better failure mode than a wrong answer.

### Provenance and the re-probe are not redundant

The easiest mistake to make here. Two of the five guardrails look like they overlap:

- **Provenance** rejects a URL that never appeared in any tool result.
- **The re-probe** independently fetches the reported URL and checks status and magic bytes.

Neither subsumes the other. A URL the model invents *and then probes itself* does enter the
discovered set — the probe result echoes it back, and every tool result is harvested — so
provenance passes for it, and only the re-probe rejects it. Conversely, a URL seen on a real page
but never probed passes provenance and is caught by the re-probe.

Small models really do invent URLs. An observed `qwen3:8b` run produced a fully plausible
`download.<host>/geoportal/rest/services/.../BD_TOPO_Batiments_France_GeoParquet.zip` at step 3.

## Nothing may hardcode a vendor, portal, or country

Beegent has to work for any `(country, use_case)` on earth, and a portal table is always one
country behind the next request. So the system prompt teaches *generic conventions* — "udata
exposes `/api/1/datasets/<slug>/`" — as shapes to try by systematic variation on a host the agent
discovered, not as a list of hostnames.

This is enforced by construction: the whole agent loop is tested against a synthetic portal for a
country that does not exist. Anything real that crept into the harness would break the tests.

## No browser, and no JavaScript execution

A Playwright-based browser existed here and was removed. A JavaScript app shell is answered by
finding the machine-readable service *behind* it, which is cheaper, more general, and yields the
download service itself rather than one file a UI happened to expose.

The cost is real: a link that only exists after a client-side interaction is unreachable. That is
reported as an honest failure. If a browser is ever re-added it should be a fourth tool the agent
escalates to, never the default path.

## The catalog stage is an empty stub

An earlier version hardcoded two catalogs and a 12-country ISO3 table. That is exactly the
country-specific knowledge the rest of the design avoids, so it was parked rather than extended.
The stage keeps its signature so the pipeline needs no branch.

## The escalation gate has one condition

Only: *was anything verified at all?*

Two broader conditions existed and were removed for the same reason — **a thin but real result is a
good outcome**. A `MIN_CANDIDATES = 2` rule escalated a single verified download. A dead-end rule
escalated a run that had verified a real national cadastre file just because two sibling angles
missed. One independently probed file is worth more than an expensive critic call spent
second-guessing it.

There is also no publisher-diversity condition. A country whose open data lives on one national
portal is a normal result, not a thin one.

## `claim` and `verification` are separate because authorship differs

The trust boundary is visible in the data rather than only in prose. `claim` is what the model
said and none of it is verified; `verification` and `cost` are what the harness measured.
`claim.file_size_bytes` and `verification.total_size_bytes` are allowed to disagree — one is a
portal's advertisement, the other is the wire.

`claim` is a whitelist rather than a copy of the model's report, so a model emitting
`{"ok": true, "payload_type": "parquet"}` cannot have those land where they read as probe results.

## Two small-model measures that look like clutter

- **History compaction** trims old tool results in place every turn. On context overflow Ollama
  evicts the *oldest* messages — the system prompt and the task — so the agent forgets its goal.
  That presents as a model failure and is a context failure.
- **The inline tool-call rescue parser** catches models that stop emitting native tool calls in
  long conversations and write JSON as text. It is parsed, executed, and the model reminded.

Both exist because the default local models actually do this.

## Angles run serially

A thread pool over the angle loop is the obvious refactor and it is wrong here for three
independent reasons: Ollama serves one request at a time by default so calls queue anyway; the
keyless search chain gets blocked when N searches leave one IP at once; and rate limits are limits
on a *rate*, so compressing the same tokens into less time multiplies the per-minute figure.

Parallelism never reduces total requests or tokens — only the window they land in, which is the
wrong direction for every dependency here. It would pay off only on a high-limit hosted endpoint
where the run is latency-bound.

## There is a connector layer, and it was rebuilt after being removed

An earlier version had one client and two base URLs, on the grounds that two OpenAI-compatible
backends did not justify an abstraction. That holds only while every backend *is* OpenAI-compatible.

The stronger argument was never extensibility: backend quirks had accumulated in shared code as
`if backend == ...` branches — a temperature retry for one endpoint, a `response_format` only one
backend accepts, content-block flattening. Each connector now owns its own. Do not remove the layer
on YAGNI grounds; that has been tried.

By contrast there is deliberately **no** pluggable layer for the web tools. Two were retired there:
an interface around an opaque `fetch_page()`, and a paid search provider whose ranking was measured
and found no better than the keyless chain. One implementation, no second one in prospect. Same
codebase, opposite answers, because the question is whether variation actually exists.

## `None` means "no answer", never "no"

`chat_json()` returning `None` means the model did not answer. Coercing that into a negative answer
silently deletes good candidates. The same principle drives the two-slot return from the fetch
stage: an angle that found nothing becomes an `unresolved` record carrying the page it reached and
the reason it gave up — excluded from candidates so nothing unverified rides in, recorded so
nothing found ever vanishes silently.
