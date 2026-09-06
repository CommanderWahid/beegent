"""Escalation gate (deterministic) + critic agent (one rare LLM call)."""

import logging

from beegent.llm import chat_json
from beegent.schemas import Candidate, SearchAngle

_log = logging.getLogger(__name__)

CRITIC_SYSTEM = """You review a failed-looking dataset discovery run and decide what happens next.

Exactly two decisions are available:
  "replan"             - there is a plausibly better angle nobody tried yet. 
                         Say concretely what to try instead.
  "needs_human_review" - the data probably does not exist in an open, findable form, or the
                         request is too vague/exotic to search for. Say why in one sentence.

Do not choose "replan" just to try again; only if you can name a genuinely different route.

Each candidate that made it through was resolved to a real file and independently probed, so
a thin result means the ANGLES were wrong, not that the verification was too strict. Angles
that dead-ended are listed too: they name portals that were reached but yielded no
downloadable file. A good replan names routes that lead to actual downloads or APIs - a
publisher's bulk-download service, a WFS/WMS/OGC endpoint, a direct file release - rather
than more pages about the data. Naming a different publisher usually beats rephrasing.

Reply with JSON only:
{"decision": "replan" | "needs_human_review", "note": "<one sentence>"}"""


def needs_escalation(candidates: list[Candidate]) -> tuple[bool, str]:
    """Deterministic threshold check - cheap, no model call."""
    # `any([])` is False, so this catches the empty run - that is what it is for.
    if not any(c.resource_url for c in candidates):
        return True, (
            "no angle resolved to a verified download"
            if not candidates
            else f"none of the {len(candidates)} candidate(s) have a fetchable resource "
            "endpoint"
        )

    return False, ""


def run_critic(
    country: str,
    use_case: str,
    angles: list[SearchAngle],
    candidates: list[Candidate],
    unresolved: list[Candidate],
    gate_reason: str,
    tried_before: list[str] | None = None,
) -> dict:
    """Run the critic LLM to decide what to do next."""
    tried = "\n".join(f"- {a.description} (channel: {a.channel_hint})" for a in angles)
    found = (
        "\n".join(
            f"- {c.url} [{c.source}, resource: {c.resource_url or 'none identified'}] "
            f"{c.title}"
            for c in candidates
        )
        or "(no angle resolved to a verified download)"
    )
    dead = (
        "\n".join(
            f"- {c.url} - {c.claim.get('failure_reason', 'no reason recorded')}"
            for c in unresolved
        )
        or "(none recorded)"
    )
    # Without this the critic cannot satisfy its own "name a genuinely different route" -
    # it has nothing to be different from, and repeats the same advice run after run.
    before = "\n".join(f"- {u}" for u in (tried_before or [])) or "(none)"
    failed = ""
    try:
        # Usage discarded - see the note in planner.py:plan().
        data, _ = chat_json(
            "critic",
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Country: {country}\nUse case: {use_case}\n\n"
                        f"Angles tried:\n{tried}\n\nVerified downloads found:\n{found}\n\n"
                        f"Angles that dead-ended:\n{dead}\n\n"
                        f"Start URLs already tried in this run and earlier ones:\n{before}"
                        f"\n\nWhy this was escalated: {gate_reason}"
                    ),
                },
            ],
        )
    except Exception as exc:
        # A throttled critic must not discard a run that already paid for its geofetch work.
        failed = f"critic call failed: {type(exc).__name__}: {exc}"
        _log.info(f"  [critic] {failed}")
        data = None
    data = data or {}

    decision = data.get("decision")
    if decision not in ("replan", "needs_human_review"):
        decision = "needs_human_review"  # unparseable critic -> fail safe, hand to a human
    # `failed` first: "the critic never answered" is not the same as "the critic escalated".
    note = failed or str(data.get("note", "")) or gate_reason
    return {"decision": decision, "note": note}
