"""CLI for the discovery-phase POC.

    python -m core.run --country Kenya --use-case "administrative boundaries for a flood dashboard"

Runs planner -> catalog workers -> geofetch (per angle) -> escalation gate -> critic,
and writes candidate_list.json either way.

Every candidate that reaches the output has had its resource_url independently probed by
core/pipeline/geofetch.py - an unverified URL cannot get here.
"""

import argparse
import json
import time

from core import config
from core.pipeline import (
    needs_escalation,
    plan,
    resolve_angle,
    run_catalog_workers,
    run_critic,
)
from core.schemas import Candidate, DiscoveryRun
from core.search_backends import normalize_url


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Order the pool and cap it. Not triage - there is no judgement here.

    Deduping is the one merge job that survives: two angles can chase different framings of
    the same dataset and land on the same file, and a re-plan can re-find what iteration 1
    already verified. First occurrence wins, and since carried candidates are passed in
    first, a re-found duplicate never displaces the one already in the list.
    """
    best: dict[str, Candidate] = {}
    for cand in candidates:
        best.setdefault(normalize_url(cand.resource_url or cand.url), cand)
    ranked = sorted(best.values(), key=lambda c: c.confidence or 0.0, reverse=True)
    return ranked[: config.MAX_FINAL_CANDIDATES]


_COST_KEYS = ("http_requests", "prompt_tokens", "completion_tokens", "total_tokens")


def discover(country: str, use_case: str) -> DiscoveryRun:
    run = DiscoveryRun(
        country=country,
        use_case=use_case,
        max_iterations=config.MAX_ITERATIONS,
        backend=config.LLM_BACKEND,
        models={
            "planner": config.PLANNER_MODEL,
            "geofetch": config.GEOFETCH_MODEL,
            "critic": config.CRITIC_MODEL,
        },
        # Accumulated as angles finish, never summed from run.candidates at the end:
        # run.unresolved is replaced each iteration, so a final sum would silently
        # under-count every dead end from iteration 1.
        totals=dict.fromkeys(("angles_run",) + _COST_KEYS, 0),
    )

    # Catalog workers are deterministic and query-independent enough to run once;
    # their results seed every planner attempt.
    print("[catalog] running catalog workers")
    catalog_hits = run_catalog_workers(country, use_case)

    feedback: str | None = None
    carried: list[Candidate] = []  # survivors from earlier iterations
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        run.iteration = iteration
        print(f"\n=== iteration {iteration}/{config.MAX_ITERATIONS} ===")

        angles = plan(country, use_case, feedback)
        print(f"[plan] {len(angles)} angle(s)")
        for angle in angles:
            print(f"  - [{angle.channel_hint}] {angle.description}")

        # Carried candidates go in first: they already resolved to a verified file, so
        # they win any collision and are never squeezed out by a re-found duplicate. A
        # re-plan must add to the pool, not restart it - otherwise a good hit from
        # iteration 1 silently vanishes because iteration 2 didn't happen to re-find it.
        if carried:
            print(f"[carry] {len(carried)} verified from iteration {iteration - 1}")

        fresh: list[Candidate] = []
        misses: list[Candidate] = []
        for i, angle in enumerate(angles, 1):
            print(f"[geofetch] angle {i}/{len(angles)}: {angle.description}")
            print(f"    [geofetch] want: {angle.dataset} [{angle.format or 'any format'}]")
            try:
                found, missed = resolve_angle(country, angle)
            except Exception as exc:  # one bad angle must not kill the run
                print(f"    [geofetch] failed: {exc}")
                found = missed = None
            if found:
                fresh.append(found)
            elif missed:
                misses.append(missed)
            # Cost is carried on whichever slot came back. When both are None the angle
            # died in its seed search, and that one request goes unrecorded - not worth
            # widening resolve_angle()'s return signature to capture.
            if found or missed:
                run.totals["angles_run"] += 1
                cost = (found or missed).cost
                for key in _COST_KEYS:
                    run.totals[key] += cost.get(key, 0)

        run.candidates = _rank(carried + list(catalog_hits) + fresh)
        run.unresolved = misses
        carried = list(run.candidates)

        escalate, gate_reason = needs_escalation(run.candidates, run.unresolved)
        if not escalate:
            run.status = "ok"
            run.reason = None
            return run

        print(f"[gate] escalating: {gate_reason}")
        if iteration >= config.MAX_ITERATIONS:
            # Hard cap is not optional - stop regardless of what the critic would say.
            run.status = "needs_human_review"
            run.reason = (
                f"{gate_reason}; reached max_iterations ({config.MAX_ITERATIONS}) "
                "without enough signal"
            )
            return run

        verdict = run_critic(country, use_case, angles, run.candidates, gate_reason)
        print(f"[critic] {verdict['decision']}: {verdict['note']}")
        if verdict["decision"] == "replan":
            feedback = verdict["note"]
            continue

        run.status = "needs_human_review"
        run.reason = verdict["note"]
        return run

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Discovery phase POC")
    parser.add_argument("--country", required=True)
    parser.add_argument("--use-case", required=True, dest="use_case")
    parser.add_argument("--out", default="candidate_list.json")
    args = parser.parse_args()

    started = time.time()
    print(
        f"[config] backend={config.LLM_BACKEND} planner={config.PLANNER_MODEL} "
        f"geofetch={config.GEOFETCH_MODEL} critic={config.CRITIC_MODEL}"
    )
    run = discover(args.country, args.use_case)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh, indent=2, ensure_ascii=False)

    print(f"\n--- summary ({time.time() - started:.0f}s) ---")
    print(f"status:     {run.status}")
    print(f"iterations: {run.iteration}/{run.max_iterations}")
    print(f"candidates: {len(run.candidates)}")
    verified = sum(1 for c in run.candidates if c.verification)
    print(f"verified:   {verified}/{len(run.candidates)} independently probed")
    if run.unresolved:
        print(f"unresolved: {len(run.unresolved)} angle(s) found no verifiable download")
        for cand in run.unresolved:
            why = str(cand.claim.get("failure_reason", ""))[:80]
            print(f"            {cand.url} - {why}")
    if run.candidates:
        top = run.candidates[0]
        print(f"top pick:   {top.title}\n            {top.url} [{top.source}]")
        print(f"            resource: {top.resource_url or 'none identified'}")
        if top.verification:
            print(
                f"            probed:   HTTP {top.verification.get('status')} "
                f"{top.verification.get('payload_type')} "
                f"({top.verification.get('first_bytes_hex')})"
            )
    if run.reason:
        print(f"reason:     {run.reason}")
    t = run.totals
    print(
        f"cost:       {t['angles_run']} angle(s), {t['http_requests']} request(s), "
        f"{t['total_tokens']:,} tokens "
        f"({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out)"
    )
    print(f"written to: {args.out}")


if __name__ == "__main__":
    main()
