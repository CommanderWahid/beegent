"""
Escalation gate (deterministic) + critic agent (one rare LLM call).
"""

from core import config
from core.llm import chat_json
from core.schemas import Candidate, SearchAngle

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


def needs_escalation(
    candidates: list[Candidate], unresolved: list[Candidate] = ()
) -> tuple[bool, str]:
    """
    Deterministic threshold check - cheap, no model call.

    Two conditions only, both about whether the PLAN worked: nothing was verified at all, or
    more angles dead-ended than paid off. Candidate count is deliberately not one of them.
    """
    # Total failure. `any([])` is False, so this is what catches an empty run - it is the
    # load-bearing branch, not the invariant guard it reads like. It doubles as that guard:
    # geofetch only returns found=True after probing, so a non-empty list reaching here with
    # no resource_url at all would mean the contract was broken upstream.
    if not any(c.resource_url for c in candidates):
        return True, (
            "no angle resolved to a verified download"
            if not candidates
            else f"none of the {len(candidates)} candidate(s) have a fetchable resource "
            "endpoint"
        )

    # More angles dead-ended than paid off: the plan itself is the problem, not this run's
    # luck. Worth a critic call before calling the result good.
    if len(unresolved) > len(candidates):
        return True, (
            f"{len(unresolved)} of {len(unresolved) + len(candidates)} angle(s) resolved "
            "to no verifiable download"
        )

    # A thin-but-real result is a pass: one independently verified download beats escalating
    # to an expensive critic call, so candidate COUNT is deliberately not a gate condition.
    return False, ""


def run_critic(
    country: str,
    use_case: str,
    angles: list[SearchAngle],
    candidates: list[Candidate],
    gate_reason: str,
) -> dict:
    """
    Run the critic LLM to decide what to do next.
    """
    tried = "\n".join(f"- {a.description} (channel: {a.channel_hint})" for a in angles)
    found = (
        "\n".join(
            f"- {c.url} [{c.source}, resource: {c.resource_url or 'none identified'}] "
            f"{c.title}"
            for c in candidates
        )
        or "(no angle resolved to a verified download)"
    )
    data = chat_json(
        config.CRITIC_MODEL,
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Country: {country}\nUse case: {use_case}\n\n"
                    f"Angles tried:\n{tried}\n\nVerified downloads found:\n{found}\n\n"
                    f"Why this was escalated: {gate_reason}"
                ),
            },
        ],
    ) or {}

    decision = data.get("decision")
    if decision not in ("replan", "needs_human_review"):
        decision = "needs_human_review"  # unparseable critic -> fail safe, hand to a human
    return {"decision": decision, "note": str(data.get("note", "")) or gate_reason}
