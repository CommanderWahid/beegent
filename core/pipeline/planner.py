"""Planner: turns a use case into 1..n distinct search angles."""

from core import config
from core.llm import chat_json
from core.schemas import SearchAngle

SYSTEM = """You plan how to find geospatial datasets on the internet.

Given a country and a use case, produce a list of distinct SEARCH ANGLES - different
routes someone would take to find suitable data. Angles must not overlap: each one
should target a different kind of publisher or a different framing of the data.

The authoritative producer is usually the country's national mapping, cadastral or geographic institute. 
Unless the use case clearly calls for something else, one angle must name that specific body for this 
country - not a generic "government data" angle, and not the national statistics office, which publishes 
statistics rather than base geometry.

Name concrete organizations wherever you can. An angle that names a publisher beats an
angle that names a category.

Decide how many angles the use case actually warrants: a narrow, well-known need may
justify only 1-2, a broad or ambiguous one up to {max_angles}. Never exceed {max_angles}.

Each angle is not just something to search for - it is a concrete FETCH TASK. A downstream
agent will chase it all the way to a file and verify the bytes, so say what file it should
come back with:

  "dataset" - the file in plain words, including its geographic extent, e.g.
              "building footprints, whole country" or "level-2 administrative boundaries".
  "format"  - the file format to ask for. Prefer, in this order: GeoParquet, GeoPackage,
              Shapefile, GeoJSON, CSV. Choose what this publisher plausibly offers,
              not always the first one. Leave it empty only if the format genuinely
              does not matter. Never ask for a documentation format (PDF, TXT, MD) -
              those are not data.
  "vintage" - "latest" unless the use case names a specific year/edition.

Reply with JSON only:
{{"angles": [{{"description": "<what to search for, phrased as a search intent>",
              "channel_hint": "catalog" | "web_search" | "national_geoportal",
              "rationale": "<one sentence on why this route is worth trying>",
              "dataset": "<the file wanted, with its extent>",
              "format": "<GeoParquet | GeoPackage | Shapefile | GeoJSON | CSV, or empty>",
              "vintage": "latest"}}]}}"""


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
                    # A missing spec must not lose the angle: description is a usable
                    # dataset description, and "" format falls back to a liveness check.
                    dataset=str(raw.get("dataset") or raw["description"]),
                    format=str(raw.get("format") or ""),
                    vintage=str(raw.get("vintage") or "latest"),
                )
            )

    if not angles:  # fallback so a bad planner call can't kill the run
        angles = [
            SearchAngle(
                description=f"{use_case} data for {country}",
                channel_hint="web_search",
                rationale="fallback: planner returned nothing usable",
                dataset=f"{use_case} for {country}",
                format="",
                vintage="latest",
            )
        ]
    return angles
