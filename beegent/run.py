"""CLI for the discovery-phase POC."""

import argparse
import json
import logging
import os
import sys
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

_log = logging.getLogger(__name__)


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Order the pool and cap it."""
    best: dict[str, Candidate] = {}
    for cand in candidates:
        best.setdefault(normalize_url(cand.resource_url or cand.url), cand)
    ranked = sorted(best.values(), key=lambda c: c.confidence or 0.0, reverse=True)
    return ranked[: config.MAX_FINAL_CANDIDATES]


_COST_KEYS = ("http_requests", "prompt_tokens", "completion_tokens", "total_tokens")
_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _add_tokens(dst: dict, usage: TokenUsage) -> None:
    """Add one usage record into any dict carrying the three token keys."""
    dst["prompt_tokens"] += usage.prompt_tokens
    dst["completion_tokens"] += usage.completion_tokens
    dst["total_tokens"] += usage.total_tokens


def _add_role(totals: dict, role: str, usage: TokenUsage) -> None:
    """Charge one role, to its by_role bucket and the run total together."""
    _add_tokens(totals["by_role"].setdefault(role, dict.fromkeys(_TOKEN_KEYS, 0)), usage)
    _add_tokens(totals, usage)


def _finalize(run: DiscoveryRun) -> DiscoveryRun:
    """Fold the per-role LLM meter into run.totals, then hand the run back."""
    for role, usage in usage_by_role().items():
        _add_role(run.totals, role, usage)
    return run


def discover(country: str, use_case: str) -> DiscoveryRun:
    connector = get_connector()
    run = DiscoveryRun(
        country=country,
        use_case=use_case,
        max_iterations=config.MAX_ITERATIONS,
        backend=connector.provider,
        models={r: connector.model_for(r) for r in connector.ROLES},
        # Accumulated as angles finish; unresolved is replaced each iteration.
        totals=dict.fromkeys(("angles_run",) + _COST_KEYS, 0) | {"by_role": {}},
    )
    # Module-level, so a second discover() would otherwise inherit the first run's tokens.
    reset_usage()

    # Deterministic and query-independent, so run once and seed every planner attempt.
    _log.info("[catalog] querying catalogs")
    catalog_hits = query_catalogs(country, use_case)

    feedback: str | None = None
    carried: list[Candidate] = []  # survivors from earlier iterations
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        run.iteration = iteration
        _log.info(f"\n=== iteration {iteration}/{config.MAX_ITERATIONS} ===")

        angles = plan(country, use_case, feedback)
        _log.info(f"[plan] {len(angles)} angle(s)")
        for angle in angles:
            _log.info(f"  - [{angle.channel_hint}] {angle.description}")

        # Carried first, so an already-verified file wins any dedupe collision.
        if carried:
            _log.info(f"[carry] {len(carried)} verified from iteration {iteration - 1}")

        fresh: list[Candidate] = []
        misses: list[Candidate] = []
        for i, angle in enumerate(angles, 1):
            _log.info(f"[geofetch] angle {i}/{len(angles)}: {angle.description}")
            try:
                found, missed = resolve_angle(angle)
            except Exception as exc:  # one bad angle must not kill the run
                _log.info(f"    [geofetch] failed: {exc}")
                found = missed = None
            if found:
                fresh.append(found)
            elif missed:
                misses.append(missed)
            # Cost rides on whichever slot came back.
            if found or missed:
                run.totals["angles_run"] += 1
                cost = (found or missed).cost
                run.totals["http_requests"] += cost.get("http_requests", 0)
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

        _log.info(f"[gate] escalating: {gate_reason}")
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
        _log.info(f"[critic] {verdict['decision']}: {verdict['note']}")
        if verdict["decision"] == "replan":
            feedback = verdict["note"]
            continue

        run.status = "needs_human_review"
        run.reason = verdict["note"]
        return _finalize(run)

    return _finalize(run)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Discovery phase POC")
    parser.add_argument("--country", required=True)
    parser.add_argument("--use-case", required=True, dest="use_case")
    parser.add_argument("--out", default="candidate_list.json")
    # Evaluated here, not at import, so a runtime-registered connector is offered too.
    parser.add_argument("--backend", choices=sorted(CONNECTORS),
                        help="LLM backend (default: $LLM_BACKEND, else ollama)")
    # One generated flag per role, so a new role gets a CLI flag for free.
    for role in LLMConnector.ROLES:
        parser.add_argument(f"--{role}-model", dest=f"{role}_model", metavar="NAME",
                            help=f"model for the {role} step "
                                 f"(default: ${role.upper()}_MODEL, else the backend's)")
    args = parser.parse_args()

    # Precedence: CLI > environment > .env > the connector's DEFAULT_MODELS.
    if args.backend:
        select_backend(args.backend)
    for role in LLMConnector.ROLES:
        value = getattr(args, f"{role}_model")
        if value:
            # The same channel model_for() reads, so precedence lives in one place.
            os.environ[f"{role.upper()}_MODEL"] = value

    connector = get_connector()
    # Fail before spending a run: a bad endpoint name otherwise 404s mid-angle.
    problem = connector.validate()
    if problem:
        raise SystemExit(f"error: {problem}")

    started = time.time()
    # repr()'d because it is free text: `use-case=''` reads unambiguously.
    _log.info(f"[input]  country={args.country}  use-case={args.use_case!r}")
    _log.info(
        f"[config] backend={connector.provider}  "
        + "  ".join(f"{r}={connector.model_for(r)}" for r in connector.ROLES)
    )
    run = discover(args.country, args.use_case)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(run.to_dict(), fh, indent=2, ensure_ascii=False)

    _log.info(f"\n--- summary ({time.time() - started:.0f}s) ---")
    _log.info(f"status:     {run.status}")
    _log.info(f"iterations: {run.iteration}/{run.max_iterations}")
    _log.info(f"candidates: {len(run.candidates)}")
    verified = sum(1 for c in run.candidates if c.verification)
    _log.info(f"verified:   {verified}/{len(run.candidates)} independently probed")
    if run.unresolved:
        _log.info(f"unresolved: {len(run.unresolved)} angle(s) found no verifiable download")
        for cand in run.unresolved:
            why = str(cand.claim.get("failure_reason", ""))[:80]
            _log.info(f"            {cand.url} - {why}")
    if run.candidates:
        top = run.candidates[0]
        _log.info(f"top pick:   {top.title}\n            {top.url} [{top.source}]")
        _log.info(f"            resource: {top.resource_url or 'none identified'}")
        if top.verification:
            _log.info(
                f"            probed:   HTTP {top.verification.get('status')} "
                f"{top.verification.get('payload_type')} "
                f"({top.verification.get('first_bytes_hex')})"
            )
    if run.reason:
        _log.info(f"reason:     {run.reason}")
    t = run.totals
    _log.info(
        f"cost:       {t['angles_run']} angle(s), {t['http_requests']} request(s), "
        f"{t['total_tokens']:,} tokens "
        f"({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out)"
    )
    # Where the tokens went, biggest spender first.
    by_role = t.get("by_role") or {}
    if by_role:
        parts = "  |  ".join(
            f"{role} {b['total_tokens']:,}"
            for role, b in sorted(by_role.items(),
                                  key=lambda kv: kv[1]["total_tokens"], reverse=True)
        )
        _log.info(f"            {parts}")
    _log.info(f"written to: {args.out}")


if __name__ == "__main__":
    main()
