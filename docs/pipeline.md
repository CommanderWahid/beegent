# Pipeline

Four chained stages, driven by the loop in `beegent/run.py:discover()`.

```
planner -> catalogs -> geofetch (per angle) -> escalation gate
        -> critic -> (replan, max 2 iterations | needs_human_review)
```

| stage | module | LLM | what it does |
|---|---|---|---|
| planner | `beegent/pipeline/planner.py` | 1 call | use case → 1..`MAX_ANGLES` distinct `SearchAngle`s |
| catalogs | `beegent/pipeline/catalogs.py` | none | **deliberately an empty stub** |
| geofetch | `beegent/pipeline/geofetch.py` | ≤20 calls per angle | chases one angle to a verified file — [details](geofetch.md) |
| rank | `beegent/run.py:_rank()` | none | dedupe, sort by confidence, cap at `MAX_FINAL_CANDIDATES` |
| gate | `beegent/pipeline/critic.py:needs_escalation()` | none | one condition: was anything verified at all? |
| critic | `beegent/pipeline/critic.py:run_critic()` | ≤1 call | `replan` or `needs_human_review` |

## An angle is a fetch order, not a search query

The planner's whole job is to map `(country, use_case)` into the four values the fetch agent is
started with, so nothing downstream has to rediscover any of them:

```python
SearchAngle(url=...,        # where to start - the deepest concrete path available
            dataset=...,    # which dataset, in free text
            format=...,     # which format is being asked for
            vintage=...)    # which edition, usually "latest"
```

`url` is the field that decides what an angle costs — see [optimizations](optimizations.md). An
angle whose `url` is missing or not `http`-prefixed is **dropped rather than half-built**; if every
angle is dropped, `plan()` returns `[]` and the run escalates to the critic, which is the machinery
working rather than a gap.

## The catalog stub

`query_catalogs()` returns `[]` and always has. The previous version hardcoded two catalogs and a
12-country ISO3 table, so it was parked rather than extended — a portal table is always one country
behind the next request. It keeps its signature so `run.py` does not need a branch.

## Re-planning adds to the pool

Iteration N+1 carries the previous verified candidates **ahead of** fresh finds, so they win
`_rank()`'s dedupe collisions. Carried candidates are never re-resolved: they already passed an
independent probe, and re-running an agent against them could only lose one to a flaky run.

`MAX_ITERATIONS` (2) is a hard cap enforced regardless of what the critic would say.

## The escalation gate has exactly one condition

```python
not any(c.resource_url for c in candidates)
```

Since `any([])` is `False`, that is also what catches an empty run.

Two other conditions existed and were **removed for the same reason** — a thin-but-real result is a
good outcome. `MIN_CANDIDATES = 2` escalated a single verified download; a dead-end rule
(`len(unresolved) > len(candidates)`) escalated a run that had verified a real national cadastre
file just because two sibling angles missed. One independently probed file is worth more than the
expensive critic call spent second-guessing it.

There is also no publisher or diversity condition. A country whose open data lives entirely on one
national portal is a normal result, not a thin one.

Because the gate fires *only* when nothing was verified, `run_critic()` sees an empty `candidates`
list essentially every time. Its real input is `unresolved` — the entry points that were reached
and the `claim.failure_reason` for each.

## Every stage fails soft

A dead catalog, a failed angle, or an unparseable critic reply must not kill the run; the critic
fails safe to `needs_human_review`. `GeofetchAgent.run()` never raises — a throttled angle degrades
to a failure record carrying its cost, rather than vanishing with no trace.

`chat_json()` returning `None` means "no answer", never a negative answer. Coercing it into a "no"
silently deletes good candidates.
