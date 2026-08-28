# Output — `candidate_list.json`

Written to `candidate_list.json` by default; override with `--out`. A run summary also goes to
stdout.

## `url` vs `resource_url`

Two different questions, and conflating them was the bug this split exists to fix — a portal
homepage used to score as high as a real dataset.

- **`url`** — the page we cite. The planner-supplied entry point the agent was started at.
- **`resource_url`** — the endpoint that actually serves data, resolved by the agent *and probed by
  the harness*.

`resource_url is None` therefore never means "not checked". It means no fetchable endpoint could be
verified — and in that case the candidate is in `unresolved`, not in `candidates`.

## The trust split

`claim` is what the model said; `verification` and `cost` are what the harness measured. **Never
let one leak into the other.**

`claim` holds the agent's own `report_result` — `edition`, `vintage_date`, `file_size_bytes`,
`checksum`, its own `high`/`medium`/`low` band, the ordered `evidence` chain, and `failure_reason`
on a dead end. **Not one key of it is verified.** Downstream code may display it, cross-check
against it, or rank on it, but must never treat it as fact.

`claim` is a **whitelist**, never a passthrough: a model that emits `{"ok": true, "payload_type":
"parquet"}` must not have those land in the output where they would read as probe results.

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
    "prompt_tokens": 182340, "completion_tokens": 4120, "total_tokens": 186460
  }
}
```

`first_bytes_hex` starting `50415231` is `PAR1` — the raw proof it really is a Parquet file.

`claim.evidence` is the audit trail: replay it to check the answer by hand. Cross-check
`claim.file_size_bytes` against `verification.total_size_bytes` — the first is what a portal page
advertised, the second is what the server actually sent. **They are allowed to disagree.**

## Confidence

One signal, only ever assigned to a verified candidate. `report.confidence` is mapped through
`config.CONFIDENCE_BY_REPORT` to `{high: 0.95, medium: 0.8, low: 0.7}`, with `CONFIDENCE_DEFAULT`
(0.7) for anything unrecognized.

The 0.7 floor is deliberate: `found=true` already means the file was independently probed and its
magic bytes matched, so even a `low` self-assessment outranks anything an unverified candidate
could ever have scored.

## `unresolved`

Angles that dead-ended: they reached a page, but no fetchable endpoint could be verified. Recorded
with `claim.failure_reason` and their own `cost` block, so a route that burned the whole step budget
is visible rather than silent. Excluded from `candidates` so nothing unverified rides in.

`cost` rides on **both** slots — a dead end that burned the entire step budget is precisely the
thing worth seeing.

## Run-level fields

`DiscoveryRun` also carries `backend`, `models` (the resolved model per role — one file tells you
what actually ran), `status`, `reason`, and `totals`:

```json
"totals": {
  "angles_run": 5, "http_requests": 28,
  "prompt_tokens": 139161, "completion_tokens": 38468, "total_tokens": 177629,
  "by_role": {
    "geofetch": {"prompt_tokens": 136851, "completion_tokens": 35920, "total_tokens": 172771},
    "planner":  {"prompt_tokens": 1840,   "completion_tokens": 2044,  "total_tokens": 3884},
    "critic":   {"prompt_tokens": 470,    "completion_tokens": 504,   "total_tokens": 974}
  }
}
```

The run-wide token counts are the sum of the `by_role` buckets — see
[optimizations](optimizations.md) for how each role is counted exactly once.
