# Output schema

A run writes `candidate_list.json` (change the path with `--out`).

```json
{
  "country": "France",
  "use_case": "building footprints as a geoparquet file",
  "iteration": 1,
  "max_iterations": 2,
  "backend": "ollama",
  "models": {"planner": "deepseek-r1:14b", "geofetch": "qwen3:8b", "critic": "deepseek-r1:14b"},
  "totals": { "...": "see below" },
  "candidates": [ "..." ],
  "unresolved": [ "..." ],
  "status": "ok",
  "reason": null
}
```

`status` is `ok` or `needs_human_review`; `reason` explains the latter.

## A candidate

```json
{
  "url": "https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_BD-TOPO",
  "title": "national mapping agency building footprints",
  "source": "geofetch",
  "confidence": 0.95,
  "resource_url": "https://data.geopf.fr/telechargement/download/.../batiment.parquet",

  "claim": {
    "edition": "BDTOPO 3.5 - 2026-06-15",
    "vintage_date": "2026-06-15",
    "file_size_bytes": 8757357612,
    "checksum": "2a463767762cffbead6122891f31d325",
    "confidence": "high",
    "evidence": [
      "Started at https://cartes.gouv.fr/… (JS app shell)",
      "Discovered data.geopf.fr/telechargement download service",
      "Located batiment.parquet on page 2 of the edition feed"
    ]
  },

  "verification": {
    "ok": true, "status": 206, "payload_type": "parquet",
    "total_size_bytes": 8757357612, "first_bytes_hex": "50415231150415e0"
  },

  "cost": {
    "steps_used": 14, "http_requests": 22,
    "prompt_tokens": 182340, "completion_tokens": 4120, "total_tokens": 186460,
    "cached_tokens": 31280
  }
}
```

### `url` vs `resource_url`

Two different things:

- **`url`** — the page to cite. The entry point the agent started from.
- **`resource_url`** — the endpoint that actually serves the data, resolved by the agent and then
  probed by the harness.

`resource_url: null` never means "not checked". It means no fetchable endpoint could be verified —
and in that case the entry is under `unresolved`, not `candidates`.

### `claim` vs `verification` — who wrote it

This is the important distinction in the whole format.

| Block | Written by | Trust |
|---|---|---|
| `claim` | the **model** | Nothing in it is verified. Display it, cross-check it, rank on it — never treat it as fact |
| `verification` | the **harness** | Measured from the wire by deterministic code |
| `cost` | the **harness** | Measured |

`claim.file_size_bytes` is what a portal page advertised. `verification.total_size_bytes` is what
the server actually sent. **They are allowed to disagree**, and a disagreement is worth looking at.

`claim` is a whitelist, not a copy of whatever the model returned, so a model cannot inject a key
that reads like a probe result.

`evidence` is usually a list of strings, but a model that emits a list of objects (`{"step": …, "url": …, "note": …}`) has its chain stored as written — it is a richer record, and flattening it would discard what the model bothered
to separate.

### Reading `verification`

| Field | Meaning |
|---|---|
| `ok` | The probe succeeded and the payload matched the requested format |
| `status` | HTTP status — `206` for a successful range request, `200` if the server ignored the range |
| `payload_type` | Identified from the first bytes: `parquet`, `zip`, `sqlite/geopackage`, `tiff/geotiff`, `pdf`, `gzip`, `7z`, `json-text`, `xml/html-text` |
| `total_size_bytes` | From `Content-Range`, when the server reports it. **`null` when `access` is `api`** — one page's length is not the dataset's size |
| `first_bytes_hex` | The raw evidence. |
| `access` | `file` or `api`. Decided by whether the reply carried service metadata, never by the model |

#### When `access` is `api`

Sixteen magic bytes can tell a Parquet file from a zip, but they cannot tell a feature service from
an error page — every JSON document starts with `{`. So a text payload gets a second, bounded read
and is parsed structurally.

| Field | Meaning |
|---|---|
| `shape` | `geojson_featurecollection`, `esrijson_featureset`, `wfs_featurecollection`, or `wfs_capabilities` |
| `feature_count` | How many features the endpoint holds — **N** |
| `count_is_exact` | `true` only when the service reported a total (`numberMatched`, `hits`, `count`). `false` means `feature_count` is a **lower bound**: the features visible in one page |
| `geometry_type` | From the first feature, in OGC names — Esri's `esriGeometryPolygon` is normalised to `Polygon` |

Always read `feature_count` together with `count_is_exact` — otherwise "the server said 40,232" and
"we counted the 1,000 that fitted in a page" are the same number in the same field. A round count
with `count_is_exact: false` is almost always a default page cap, not a small dataset.

### `confidence`

A float, only ever set on a verified candidate. It maps the model's own `high`/`medium`/`low`
self-assessment to `0.95` / `0.8` / `0.7`.

The floor of 0.7 is deliberate: a candidate only gets a confidence at all after its bytes were
checked, so even a `low` self-assessment describes a file that demonstrably exists in the right
format.

## `unresolved`

Angles that reached a page but produced no verifiable download:

```json
{
  "url": "https://www.ign.fr/contacter",
  "title": "national geoportal building data",
  "source": "geofetch",
  "confidence": null,
  "resource_url": null,
  "claim": {"failure_reason": "gave up: 3 repeated tool calls, the agent was not converging"},
  "cost": {"steps_used": 13, "http_requests": 8, "total_tokens": 38163}
}
```

Dead ends carry their `cost`, so a route that burned the whole budget is visible instead of silent.

## `totals`

```json
"totals": {
  "angles_run": 5, "http_requests": 28,
  "prompt_tokens": 139161, "completion_tokens": 38468, "total_tokens": 177629,
  "cached_tokens": 24310,
  "by_role": {
    "geofetch": {"prompt_tokens": 136851, "completion_tokens": 35920,
                 "total_tokens": 172771, "cached_tokens": 24310},
    "planner":  {"prompt_tokens": 1840, "completion_tokens": 2044,
                 "total_tokens": 3884, "cached_tokens": 0},
    "critic":   {"prompt_tokens": 470, "completion_tokens": 504,
                 "total_tokens": 974, "cached_tokens": 0}
  }
}
```

The run-wide token counts are the sum of the `by_role` buckets. See
[Performance](performance.md) for what to do with these numbers.

`cached_tokens` is a **subset of `prompt_tokens`**, not an addition to it — it is the part of the
input a backend served from its own prompt cache, so it is never included in `total_tokens`. Read it
as a ratio against `prompt_tokens`: on a backend that caches, a high ratio means most of each
request cost nothing, which matters most where a rate limit is measured in tokens per minute.
