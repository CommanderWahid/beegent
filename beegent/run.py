"""CLI for the discovery-phase POC.

    python -m beegent.run --country Kenya --use-case "administrative boundaries for a flood dashboard"

Runs planner -> catalogs -> geofetch (per angle) -> escalation gate -> critic,
and writes candidate_list.json either way.

Every candidate that reaches the output has had its resource_url independently probed by
beegent/pipeline/geofetch.py - an unverified URL cannot get here.
"""

import argparse
import json
import os
import time

from beegent import config
from beegent.connectors import CONNECTORS, LLMConnector
from beegent.llm import get_connector, reset_usage, select_backend, usage_by_role
from beegent.pipeline import (
    needs_escalation,
    plan,
    resolve_angle,
    query_catalogs,
    run_critic,
)
from beegent.schemas import Candidate, DiscoveryRun, TokenUsage
from beegent.web_tools import normalize_url


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
_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _add_role(totals: dict, role: str, usage: TokenUsage) -> None:
    """Add one role's tokens to its bucket in totals["by_role"]."""
    bucket = totals["by_role"].setdefault(role, dict.fromkeys(_TOKEN_KEYS, 0))
    bucket["prompt_tokens"] += usage.prompt_tokens
    bucket["completion_tokens"] += usage.completion_tokens
    bucket["total_tokens"] += usage.total_tokens


def _finalize(run: DiscoveryRun) -> DiscoveryRun:
    """Fold the per-role LLM meter into run.totals, then hand the run back.

    EVERY exit from discover() returns through here - including the early "ok" one, which
    is the common case - so planner and critic spend is recorded whichever branch ended the
    run. Miss one and the cheapest, most frequent path is the one that under-reports.

    Geofetch is not in the meter: its tokens are already in totals, accumulated per angle
    from Candidate.cost. That split is what keeps nothing counted twice.
    """
    for role, usage in usage_by_role().items():
        _add_role(run.totals, role, usage)
        run.totals["prompt_tokens"] += usage.prompt_tokens
        run.totals["completion_tokens"] += usage.completion_tokens
        run.totals["total_tokens"] += usage.total_tokens
    return run


def discover(country: str, use_case: str) -> DiscoveryRun:
    run = DiscoveryRun(
        country=country,
        use_case=use_case,
        max_iterations=config.MAX_ITERATIONS,
        backend=get_connector().provider,
        models={r: get_connector().model_for(r) for r in get_connector().ROLES},
        # Accumulated as angles finish, never summed from run.candidates at the end:
        # run.unresolved is replaced each iteration, so a final sum would silently
        # under-count every dead end from iteration 1.
        totals=dict.fromkeys(("angles_run",) + _COST_KEYS, 0) | {"by_role": {}},
    )
    # The meter is module-level, so a second discover() in one process would otherwise
    # inherit the first run's planner and critic tokens.
    reset_usage()

    # Catalog workers are deterministic and query-independent enough to run once;
    # their results seed every planner attempt.
    print("[catalog] querying catalogs")
    catalog_hits = query_catalogs(country, use_case)

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
            try:
                found, missed = resolve_angle(angle)
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
                _add_role(run.totals, "geofetch",
                          TokenUsage(prompt_tokens=cost.get("prompt_tokens", 0),
                                     completion_tokens=cost.get("completion_tokens", 0)))

        run.candidates = _rank(carried + list(catalog_hits) + fresh)
        run.unresolved = misses
        carried = list(run.candidates)

        escalate, gate_reason = needs_escalation(run.candidates)
        if not escalate:
            run.status = "ok"
            run.reason = None
            return _finalize(run)

        print(f"[gate] escalating: {gate_reason}")
        if iteration >= config.MAX_ITERATIONS:
            # Hard cap is not optional - stop regardless of what the critic would say.
            run.status = "needs_human_review"
            run.reason = (
                f"{gate_reason}; reached max_iterations ({config.MAX_ITERATIONS}) "
                "without enough signal"
            )
            return _finalize(run)

        verdict = run_critic(
            country, use_case, angles, run.candidates, run.unresolved, gate_reason
        )
        print(f"[critic] {verdict['decision']}: {verdict['note']}")
        if verdict["decision"] == "replan":
            feedback = verdict["note"]
            continue

        run.status = "needs_human_review"
        run.reason = verdict["note"]
        return _finalize(run)

    return _finalize(run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discovery phase POC")
    parser.add_argument("--country", required=True)
    parser.add_argument("--use-case", required=True, dest="use_case")
    parser.add_argument("--out", default="candidate_list.json")
    # choices is evaluated here, not at import, so a connector registered at runtime by a
    # wrapper script is offered too.
    parser.add_argument("--backend", choices=sorted(CONNECTORS),
                        help="LLM backend (default: $LLM_BACKEND, else ollama)")
    # One flag per pipeline role, generated rather than written out, so adding a role to
    # LLMConnector.ROLES gives it a CLI flag for free - the same way ROLES already drives
    # the banner and DiscoveryRun.models.
    for role in LLMConnector.ROLES:
        parser.add_argument(f"--{role}-model", dest=f"{role}_model", metavar="NAME",
                            help=f"model for the {role} step "
                                 f"(default: ${role.upper()}_MODEL, else the backend's)")
    args = parser.parse_args()

    # Precedence: CLI > environment > .env > the connector's DEFAULT_MODELS. The first two
    # are applied here; load_dotenv(override=False) is what puts .env below the environment.
    if args.backend:
        select_backend(args.backend)
    for role in LLMConnector.ROLES:
        value = getattr(args, f"{role}_model")
        if value:
            # Deliberately the same channel model_for() already reads, so precedence lives
            # in one place instead of being re-implemented at the CLI layer.
            os.environ[f"{role.upper()}_MODEL"] = value

    connector = get_connector()
    # Fail before spending a run on a bad endpoint: a wrong serving-endpoint name otherwise
    # surfaces as a 404 partway through an angle, after tokens are already gone.
    problem = connector.validate()
    if problem:
        raise SystemExit(f"error: {problem}")

    started = time.time()
    # What was asked, then what will answer it. The use case is repr()'d because it is free
    # text: `use-case=''` reads unambiguously where a bare `use-case=` would not.
    print(f"[input]  country={args.country}  use-case={args.use_case!r}")
    print(
        f"[config] backend={connector.provider}  "
        + "  ".join(f"{r}={connector.model_for(r)}" for r in connector.ROLES)
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
    # Where the tokens went. Biggest spender first - on a run that found nothing this is
    # how you see whether the planner or the critic was the one burning them.
    by_role = t.get("by_role") or {}
    if by_role:
        parts = "  |  ".join(
            f"{role} {b['total_tokens']:,}"
            for role, b in sorted(by_role.items(),
                                  key=lambda kv: kv[1]["total_tokens"], reverse=True)
        )
        print(f"            {parts}")
    print(f"written to: {args.out}")


if __name__ == "__main__":
    main()
