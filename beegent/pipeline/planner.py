"""Planner: turns a use case into 1..n distinct search angles."""

from beegent import config
from beegent.llm import chat_json
from beegent.schemas import SearchAngle

SYSTEM = """You plan how to find geospatial datasets on the internet.

Given a country and a use case, produce a list of distinct SEARCH ANGLES - different
routes someone would take to find suitable data. Angles must not overlap: each one
should target a different kind of publisher or a different framing of the data.

The authoritative producer is usually the country's national mapping, cadastral or geographic institute. 
Unless the use case clearly calls for something else, one angle must name that specific body for this 
country - not a generic "government data" angle, and not the national statistics office, which publishes 
statistics rather than base geometry.

Name concrete organizations AND concrete paths wherever you can. An angle that names a
publisher beats one that names a category, and an angle that names a file server beats one
that names a publisher's front door.

If the use case names a file format, that format is non-negotiable and goes on EVERY angle,
even for a publisher you think does not offer it. Finding out is the downstream agent's job -
it may convert, find a mirror, or fail honestly, and an honest failure is more useful than
silently fetching something the user did not ask for.

Decide how many angles the use case actually warrants: a narrow, well-known need may
justify only 1-2, a broad or ambiguous one up to {max_angles}. Never exceed {max_angles}.

Each angle is a concrete FETCH TASK, not a search query. A downstream agent is started with
exactly the four values you give below and chases them to a file, verifying the bytes. Your
job is to produce those four - it cannot rediscover them:

  "url"     - THE most important field you produce, and the one that decides whether this
              angle costs 16k tokens or 200k. Give the deepest concrete path you can name.

              WANT: a bulk or directory file server (.../data/latest/, .../telechargement/),
              an Atom/STAC/OGC/CSW endpoint, or an API base that takes query parameters.
              These answer the question directly - the listing IS the data.

              AVOID: a portal search page, a dataset landing page, a repository index
              (e.g. a bare .../repositories/), or a publisher homepage. Measured: these take
              about 4x the steps at 3x the payload per step and usually find nothing,
              because the agent must first discover the download service before it can even
              begin looking for the file.

              Being wrong is cheap - the agent fetches it, can search from there, and the
              bytes are verified either way. So guess deep rather than playing safe: an
              angle with no URL is discarded entirely.
  "dataset" - the file in plain words, including its geographic extent, e.g.
              "building footprints, whole country" or "level-2 administrative boundaries".
  "format"  - the file format to ask for. If the use case named one, use exactly that on
              every angle - see above, it overrides everything in this paragraph. Only
              when the use case is silent do you choose: prefer, in this order,
              GeoParquet, GeoPackage, Shapefile, GeoJSON, CSV, picking what this
              publisher plausibly offers rather than always the first. Leave it empty
              only if the format genuinely does not matter. Never ask for a
              documentation format (PDF, TXT, MD) - those are not data.
  "vintage" - "latest" unless the use case names a specific year/edition.

Reply with JSON only:
{{"angles": [{{"description": "<what to search for, phrased as a search intent>",
              "channel_hint": "catalog" | "web_search" | "national_geoportal",
              "rationale": "<one sentence on why this route is worth trying>",
              "url": "https://<host>/<path>",
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

    # Usage discarded: llm.chat_json already recorded it against the role.
    data, _ = chat_json(
        "planner",
        [
            {"role": "system", "content": SYSTEM.format(max_angles=config.MAX_ANGLES)},
            {"role": "user", "content": user},
        ],
    )

    angles: list[SearchAngle] = []
    for raw in (data or {}).get("angles", [])[: config.MAX_ANGLES]:
        if not (isinstance(raw, dict) and raw.get("description")):
            continue
        url = str(raw.get("url") or "").strip()
        if not url.startswith("http"):
            # No start URL means no fetch task, so drop it rather than half-build it.
            print(f"  [plan] dropping angle with no usable url: {raw['description'][:60]}")
            continue
        angles.append(
            SearchAngle(
                description=str(raw["description"]),
                channel_hint=str(raw.get("channel_hint", "web_search")),
                rationale=str(raw.get("rationale", "")),
                url=url,
                # A missing spec must not lose the angle.
                dataset=str(raw.get("dataset") or raw["description"]),
                format=str(raw.get("format") or ""),
                vintage=str(raw.get("vintage") or "latest"),
            )
        )
    return angles
