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

Each candidate lists whether a fetchable resource endpoint was identified for it. When the run
failed because nothing was fetchable, a good replan names routes that lead to actual downloads
or APIs - a data portal's export/API endpoints, a WFS/WMS service, a direct file release - rather
than more pages about the data.

Reply with JSON only:
{"decision": "replan" | "needs_human_review", "note": "<one sentence>"}"""


def needs_escalation(
    candidates: list[Candidate], unresolved: list[Candidate] = ()
) -> tuple[bool, str]:
    """
    Deterministic threshold check - cheap, no model call.

    TODO: tune these numbers once we have run data.
    """
    if len(candidates) < config.MIN_CANDIDATES:
        return True, f"only {len(candidates)} candidate(s) survived triage"

    # Fetchability is not negotiable: a run can be diverse, plentiful and topically perfect and
    # still be worthless if not one candidate has an endpoint you can actually pull data from.
    if not any(c.resource_url for c in candidates):
        return True, (
            f"none of the {len(candidates)} candidate(s) have a fetchable resource endpoint"
        )

    # A run that failed to judge most of what it found has not earned a confident "ok".
    if len(unresolved) > len(candidates):
        return True, (
            f"triage could not judge {len(unresolved)} of "
            f"{len(unresolved) + len(candidates)} candidates"
        )

    publishers = {c.publisher_key() for c in candidates}
    if len(publishers) == 1 and len(candidates) < config.SINGLE_PUBLISHER_MIN_CANDIDATES:
        return True, (
            f"only {len(candidates)} candidate(s), all from a single publisher "
            f"({candidates[0].publisher or candidates[0].publisher_key()})"
        )
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
            f"- {c.url} [{c.source}, publisher: {c.publisher or 'unknown'}, "
            f"resource: {c.resource_url or 'none identified'}] {c.title}"
            for c in candidates
        )
        or "(nothing survived triage)"
    )
    data = chat_json(
        config.CRITIC_MODEL,
        [
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Country: {country}\nUse case: {use_case}\n\n"
                    f"Angles tried:\n{tried}\n\nWhat triage produced:\n{found}\n\n"
                    f"Why this was escalated: {gate_reason}"
                ),
            },
        ],
    ) or {}

    decision = data.get("decision")
    if decision not in ("replan", "needs_human_review"):
        decision = "needs_human_review"  # unparseable critic -> fail safe, hand to a human
    return {"decision": decision, "note": str(data.get("note", "")) or gate_reason}
