"""CLI for the discovery-phase POC.

    python run.py --country Kenya --use-case "administrative boundaries for a flood dashboard"

Runs planner -> catalog workers -> search & explore -> merge/triage -> escalation
gate -> critic, and writes candidate_list.json either way.
"""

import argparse
import json
import time

import config
from catalog_workers import run_catalog_workers
from critic import needs_escalation, run_critic
from merge_triage import merge_and_triage
from planner import plan
from schemas import Candidate, DiscoveryRun
from search_explore import explore_angle


def discover(country: str, use_case: str) -> DiscoveryRun:
    run = DiscoveryRun(
        country=country, use_case=use_case, max_iterations=config.MAX_ITERATIONS
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

        # Carried candidates go in first: they already passed triage, so they should
        # win any dedupe collision and never be squeezed out by the per-domain cap.
        # A re-plan must add to the pool, not restart it - otherwise a good hit from
        # iteration 1 silently vanishes because iteration 2 didn't happen to re-find it.
        # Unresolved ones ride along too: their confidence is None, so a re-plan gives
        # them the real triage call they never got.
        raw = carried + list(run.unresolved) + list(catalog_hits)
        if carried or run.unresolved:
            print(
                f"[carry] {len(carried)} carried + {len(run.unresolved)} unresolved "
                f"from iteration {iteration - 1}"
            )
        for i, angle in enumerate(angles, 1):
            print(f"[explore] angle {i}/{len(angles)}: {angle.description}")
            try:
                found = explore_angle(country, use_case, angle)
            except Exception as exc:  # one bad angle must not kill the run
                print(f"    [explore] failed: {exc}")
                found = []
            print(f"    [explore] {len(found)} candidate(s)")
            raw.extend(found)

        run.candidates, run.unresolved = merge_and_triage(country, use_case, raw)
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
        f"search={config.SEARCH_EXPLORE_MODEL} triage={config.TRIAGE_MODEL} "
        f"critic={config.CRITIC_MODEL}"
    )
    run = discover(args.country, args.use_case)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh, indent=2, ensure_ascii=False)

    print(f"\n--- summary ({time.time() - started:.0f}s) ---")
    print(f"status:     {run.status}")
    print(f"iterations: {run.iteration}/{run.max_iterations}")
    print(f"candidates: {len(run.candidates)}")
    fetchable = sum(1 for c in run.candidates if c.resource_url)
    print(f"fetchable:  {fetchable}/{len(run.candidates)} with a resource endpoint")
    if run.unresolved:
        print(f"unresolved: {len(run.unresolved)} (triage gave no verdict - see JSON)")
        for cand in run.unresolved:
            print(f"            {cand.url}")
    if run.candidates:
        top = run.candidates[0]
        print(f"top pick:   {top.title}\n            {top.url} [{top.source}]")
        print(f"            resource: {top.resource_url or 'none identified'}")
    if run.reason:
        print(f"reason:     {run.reason}")
    print(f"written to: {args.out}")


if __name__ == "__main__":
    main()
