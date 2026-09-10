"""The front door: classify a request, gate a data run on country + use case."""

import pycountry

from beegent.llm import chat_json

GREETING = (
    "I find geospatial data and verify it actually downloads.\n\n"
    "To search I need a **use case** and a **country**. Tell me a **format** "
    "(GeoPackage, GeoParquet, Shapefile, GeoJSON — or **API** for a live service rather "
    "than a file) and a **vintage** if they matter — otherwise I take whatever the "
    "publisher serves, newest first.\n\n"
    "You can also ask what I already hold."
)

SYSTEM = """You are the front door to a geospatial data discovery pipeline. You do not search
the web and you have no tools. Classify the user's message and extract what they gave you.

"intent" is one of:
  "search"  - they want data fetched. This costs real money, so it needs a country.
  "catalog" - they are asking what is already stored: which links, which countries, what a
              previous run found, what has stopped working.
  "other"   - anything else, including greetings.

Extract only what the user actually said. Rules that matter:
  - NEVER invent a country. If they named a region ("Europe"), a continent, or nothing at
    all, leave "country" empty and ask.
  - NEVER invent a format. A user naming a CHANNEL ("an API", "WFS", "OGC API-Features")
    has not named a format - leave it empty, because a service serves whichever format you
    ask it for, and guessing one filters out the publisher's actual distribution.
  - A channel request sets "channel" to "api" instead - that is where "an API", "a service",
    "WFS" or "OGC API-Features" belongs. Anything else leaves "channel" empty.
  - NEVER produce a URL, a dataset id, or a publisher name. You have not read one.
  - "vintage" is "latest" unless they named a year or edition.

Reply with JSON only:
{"intent": "search"|"catalog"|"other",
 "country": "<as the user wrote it, or empty>",
 "use_case": "<what data they want, in their words, or empty>",
 "format": "<only if they named one>",
 "channel": "api" if they asked for a live service, else empty,
 "vintage": "<only if they named one>",
 "reply": "<one or two sentences to the user>"}"""


#: A channel-shaped phrase the planner is taught to recognise and leave `format` empty for.
API_PHRASE = "from an OGC API-Features or WFS endpoint"


def _still_needed(missing: list[str], data: dict) -> str:
    """What blocks the run, in the harness's words - the model cannot know it was gated."""
    if "country" in missing:
        raw = str(data.get("country") or "").strip()
        said = f"I could not match {raw!r} to a country" if raw else "I still need a country"
        which = " and a use case" if "use_case" in missing else ""
        return f"{said}{which} — which country should I search? It has to be a single country."
    return "I still need to know what data you are after."


def compose_use_case(fields: dict) -> str:
    """Fold the four fields into ONE string - discover() reads nothing else."""
    use_case = str(fields.get("use_case") or "").strip()
    if not use_case:
        return ""
    low = use_case.lower()
    fmt = str(fields.get("format") or "").strip()
    vintage = str(fields.get("vintage") or "").strip()
    out = use_case
    if fields.get("channel") == "api" and not fmt:
        if "api" not in low and "wfs" not in low and "service" not in low:
            out += f" {API_PHRASE}"
    elif fmt and fmt.lower() not in low:
        out += f" as a {fmt} file"
    if vintage and vintage.lower() != "latest" and vintage.lower() not in low:
        out += f", {vintage} edition"
    return out


def real_country(name: str) -> str:
    """The canonical name, or empty. A standard, never a hand-kept list of places."""
    name = name.strip()
    if not name:
        return ""
    try:
        return pycountry.countries.lookup(name).name
    except LookupError:
        pass
    # "Netherland" is a typo, not a region: fuzzy catches it, and still finds nothing for
    # "Europe" or "Atlantis". The hit is a SUGGESTION - the confirmation card shows it first.
    if len(name) < 4:
        return ""  # a real code already matched above; a short string only invites a wrong hit
    try:
        return pycountry.countries.search_fuzzy(name)[0].name
    except LookupError:
        return ""


def interpret(message: str, history: list[dict] | None = None) -> dict:
    """One LLM call. Returns the extracted fields plus what still blocks a run."""
    messages = [{"role": "system", "content": SYSTEM}]
    messages += history or []
    messages.append({"role": "user", "content": message})
    try:
        data, _ = chat_json("planner", messages)
    except Exception as exc:  # every stage fails soft; the front door is no exception
        return {"intent": "other", "missing": [],
                "reply": f"I could not read that ({type(exc).__name__}). Try again?"}
    data = data or {}

    fields = {k: str(data.get(k) or "").strip()
              for k in ("country", "use_case", "format", "vintage")}
    # A channel is not a format: it steers the wording, it never pins the payload guardrail.
    fields["channel"] = "api" if str(data.get("channel") or "").lower() == "api" else ""
    # "Present" is not "valid": a region or a typo resolves to nothing and is asked about.
    fields["country"] = real_country(fields["country"])
    intent = data.get("intent") if data.get("intent") in ("search", "catalog", "other") else "other"

    missing = [f for f in ("country", "use_case") if not fields[f]] if intent == "search" else []
    out = {"intent": intent, "missing": missing,
           "ready": intent == "search" and not missing,
           "reply": str(data.get("reply") or ""), **fields}
    if missing:
        # The model writes "Got it" believing it had a country; the gate disagreed, and
        # only the harness knows that. Say so, or the UI promises work it will not do.
        out["reply"] = (out["reply"] + "\n\n" + _still_needed(missing, data)).strip()
    # What the run will actually be given, so the confirmation shows the real request.
    out["sent_as"] = compose_use_case(out) if out["ready"] else ""
    return out
