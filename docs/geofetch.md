# The geofetch agent

Geofetch is the stage that does the work.<br>
One tool-calling agent runs per angle, and it is the only part of Beegent that touches the web.<br><br>

```mermaid
flowchart TD
    ANGLE["SearchAngle from the planner<br/>url + dataset + format + vintage"]

    subgraph AGENT["agent loop - up to GEOFETCH_MAX_STEPS (20)"]
        LLM["chat_tools() · generic methodology prompt:<br/>recon, search early, platform API conventions,<br/>mine urls_found, compare vintages, never invent URLs"]
        COMPACT["compact history each turn<br/>old tool results trimmed in place"]
        RESCUE["rescue parser<br/>a tool call written as plain text<br/>is parsed and executed anyway"]
        ANTILOOP{"exact call<br/>already made?"}
        WARN["REPEATED CALL warning<br/>+ task reminder · no HTTP spent"]
        T1["fetch_page<br/>text+links, or raw XML/JSON<br/>urls_found = URLs no anchor carries"]
        T2["web_search<br/>ddg-html, ddg-lite, bing<br/>bot-block detection"]
        T3["probe_url<br/>Range bytes=0-15<br/>status, size, magic bytes"]
        COMPACT --> LLM
        LLM --> ANTILOOP
        RESCUE --> LLM
        ANTILOOP -->|repeat| WARN --> LLM
        ANTILOOP -->|new| T1 --> LLM
        ANTILOOP -->|new| T2 --> LLM
        ANTILOOP -->|new| T3 --> LLM
    end

    ANGLE --> LLM

    WEB["the open web<br/>portals, catalogues, APIs<br/>Atom, DCAT, CKAN, udata, GeoNetwork, STAC, OGC"]
    T1 --- WEB
    T2 --- WEB
    T3 --- WEB

    G{"guardrail chain, in order<br/>1 gave up before 5 requests, bounce once<br/>2 found=true with no download_url<br/>3 URL never seen in a tool result, invented<br/>4 independent re-probe fails or serves HTML<br/>5 magic bytes do not match the requested format<br/>6 text that is no feature collection, or a capabilities doc<br/>7 a service returning zero features"}
    LLM -->|report_result| G
    G -->|any check fails: REPORT REJECTED| LLM

    OKC["Candidate<br/>resource_url + claim + verification + cost"]
    BADC["unresolved Candidate<br/>claim.failure_reason + cost"]
    G -->|all checks pass| OKC
    G -->|honest failure| BADC
```

## The three tools

| Tool | Arguments | Returns |
|---|---|---|
| `fetch_page` | `url`, optional `accept` | HTML reduced to text and links; XML/JSON returned raw and truncated. Always includes `urls_found` |
| `web_search` | `query` | Results from a DuckDuckGo HTML → DuckDuckGo Lite → Bing chain, with bot-block detection |
| `probe_url` | `url` | HTTP status, total size, and the payload type identified from the first 16 bytes |


## What the agent is told

The system prompt teaches **method, not examples**: do reconnaissance first, search the web early,
try standard catalogue API conventions, mine urls, filter server-side rather than scraping, compare explicit edition dates when asked for "latest", never invent a URL, and verify before reporting.

Nothing in it names a real portal. That is what lets the same prompt work for any country.

## No browser, no JavaScript

Beegent runs no headless browser and executes no JavaScript. A JS app shell is not rendered — it is
routed *around*.

| | |
|---|---|
| **Handled** | The machine-readable service behind the shell — a `<link rel=alternate>` hint, a platform API convention, or an API base mined. |
| **Not handled** | A download URL built by a client-side interaction. It appears in no served byte, so there is nothing to discover — reported as an honest failure, never guessed |

## The seven guardrails

The model can only finish by calling `report_result`. The harness then checks the report with
deterministic code before accepting it:

| # | Check | Rejects |
|---|---|---|
| 1 | Minimum effort | A give-up filed before 5 HTTP requests — bounced once with a checklist of untried techniques |
| 2 | URL present | `found: true` with no `download_url` |
| 3 | Provenance | A URL that never appeared in any tool result — invented, rejected even if it is live |
| 4 | Independent re-probe | A bad status, or an HTML error page served at a download URL |
| 5 | Format match | A live file of the wrong type — a GeoJSON cannot satisfy a GeoParquet request |
| 6 | Feature service is data | A capabilities document, or text that parses as no feature collection at all — some services answer errors with HTTP 200 |
| 7 | Non-empty result | A live, well-formed endpoint returning **zero** features. The API analogue of check 5 |

<br>A rejected report is handed back to the model, which must keep searching. **A fabricated URL cannot
reach the output; the worst case is an honest failure** recorded under `unresolved`.


## When the agent gets stuck

- **Repeated calls.** After `GEOFETCH_MAX_REPEATS` the angle is abandoned.
- **Budget exhausted.** When `GEOFETCH_MAX_HTTP_REQUESTS` is spent the angle stops immediately —
  every further tool call would only error, so no more LLM calls are made. Verifying a download
  needs a probe, so an angle that reaches this point cannot report a result.
- **Context pressure.** Old tool results are trimmed in place every turn so the system prompt and
  the task survive on small local context windows.
- **Malformed tool calls.** Some models stop emitting native tool calls in long conversations and
  write JSON as plain text. That is parsed, executed anyway, and the model reminded.

## Known limitations

- **No JavaScript.** A download URL that exists only after a client-side interaction appears in no
  served byte. Reported as an honest failure — see the table above.
- **Coordinate reference systems are never inspected.** The agent verifies *bytes*, not geometry.
  The case that matters: China mandates the obfuscated GCJ-02 datum, so data from Chinese sources
  is shifted by a few hundred metres relative to WGS84. The file verifies, the geometry parses, and
  the coordinates are wrong. There is no fix inside a byte-level probe, and detecting it by host
  country would mean hardcoding a country — which this codebase does not do anywhere.
- **`feature_count` may be a lower bound.** See `count_is_exact` in
  [the output schema](output-schema.md).

## Related

- [Configuration](configuration.md) — the step, request and truncation limits
- [Performance](performance.md) — what each limit costs
