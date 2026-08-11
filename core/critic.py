"""Escalation gate (deterministic) + critic agent (one rare LLM call)."""

import config
from llm import chat_json
from schemas import Candidate, SearchAngle

CRITIC_SYSTEM = """You review a failed-looking dataset discovery run and decide what happens next.

Exactly two decisions are available:
  "replan"             - there is a plausibly better angle nobody tried yet. Say concretely
                         what to try instead (e.g. "try the national statistics office and
                         the national geoportal instead of generic web search").
  "needs_human_review" - the data probably does not exist in an open, findable form, or the
                         request is too vague/exotic to search for. Say why in one sentence.

Do not choose "replan" just to try again; only if you can name a genuinely different route.

Reply with JSON only:
{"decision": "replan" | "needs_human_review", "note": "<one sentence>"}"""


def needs_escalation(
    candidates: list[Candidate], unresolved: list[Candidate] = ()
) -> tuple[bool, str]:
    """Deterministic threshold check - cheap, no model call.

    Diversity is measured on the PUBLISHER triage guessed, not on Candidate.source
    (the worker that found it) and not on the raw domain. Which worker found a
    candidate says nothing about its quality - catalog workers coming up empty is
    normal for use cases HDX simply doesn't cover - and raw domain would flag a
    strong national portal hosting several agencies' datasets as undiverse.

    A single publisher is only a failure signal when the result is ALSO thin: a
    handful of distinct, high-confidence datasets from one good publisher is a
    fine outcome and must not escalate on its own.

    TODO: tune these numbers once we have run data.
    """
    if len(candidates) < config.MIN_CANDIDATES:
        return True, f"only {len(candidates)} candidate(s) survived triage"

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
    tried = "\n".join(f"- {a.description} (channel: {a.channel_hint})" for a in angles)
    found = (
        "\n".join(
            f"- {c.url} [{c.source}, publisher: {c.publisher or 'unknown'}] {c.title}"
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
