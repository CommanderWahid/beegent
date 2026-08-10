"""Planner: turns a use case into 1..n distinct search angles."""

import config
from llm import chat_json
from schemas import SearchAngle

SYSTEM = """You plan how to find geospatial datasets on the internet.

Given a country and a use case, produce a list of distinct SEARCH ANGLES - different
routes someone would take to find suitable data. Angles must not overlap: each one
should target a different kind of publisher or a different framing of the data.

Decide how many angles the use case actually warrants: a narrow, well-known need may
justify only 1-2, a broad or ambiguous one up to {max_angles}. Never exceed {max_angles}.

Reply with JSON only:
{{"angles": [{{"description": "<what to search for, phrased as a search intent>",
              "channel_hint": "catalog" | "web_search" | "national_geoportal",
              "rationale": "<one sentence on why this route is worth trying>"}}]}}"""


def plan(country: str, use_case: str, feedback: str | None = None) -> list[SearchAngle]:
    user = f"Country: {country}\nUse case: {use_case}"
    if feedback:
        user += (
            "\n\nA previous attempt produced too little usable signal. "
            f"A reviewer said to try differently: {feedback}\n"
            "Produce new angles that follow that advice; do not repeat what clearly failed."
        )

    data = chat_json(
        config.PLANNER_MODEL,
        [
            {"role": "system", "content": SYSTEM.format(max_angles=config.MAX_ANGLES)},
            {"role": "user", "content": user},
        ],
    )

    angles: list[SearchAngle] = []
    for raw in (data or {}).get("angles", [])[: config.MAX_ANGLES]:
        if isinstance(raw, dict) and raw.get("description"):
            angles.append(
                SearchAngle(
                    description=str(raw["description"]),
                    channel_hint=str(raw.get("channel_hint", "web_search")),
                    rationale=str(raw.get("rationale", "")),
                )
            )

    if not angles:  # fallback so a bad planner call can't kill the run
        angles = [
            SearchAngle(
                description=f"{use_case} data for {country}",
                channel_hint="web_search",
                rationale="fallback: planner returned nothing usable",
            )
        ]
    return angles
