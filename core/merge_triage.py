"""Merge & triage: dedupe, drop obvious junk for free, then classify what survives.

Two layers, cheap first - the deterministic pre-filter costs nothing and kills the
easy junk before any model call.
"""

import mimetypes
from urllib.parse import urlparse

import config
from llm import chat_json
from schemas import Candidate

TRIAGE_SYSTEM = """You judge whether a URL is a usable geospatial DATA SOURCE for a
specific request - not merely a page that mentions the topic.

Set is_data_source true only if the page is (or directly serves) a dataset, download,
API or data catalog entry that plausibly covers the requested country and use case.
Set it false for blog posts, news, tutorials, product marketing, generic homepages,
social media, forums and Q&A threads, encyclopedia articles, search-engine result pages,
and for data that is about something else entirely.

confidence answers a DIFFERENT question from is_data_source. It is not how sure you are
of the true/false call - it is how DIRECTLY this resource serves the requested country
and use case. Score it against these bands:

  0.9-1.0  a specific named dataset, download or API that matches both the country and
           the use case, with no reservations
  0.7-0.8  a real dataset page, but broader or adjacent - a national theme layer that
           contains what was asked for, a superset, or one region of the country
  0.5-0.6  a catalog or portal entry that probably leads to the data but is not the data
           itself; or the coverage is plausible but you cannot verify it
  0.1-0.4  mentions the topic but is not itself a data source (set is_data_source false)

The URL itself is your main evidence for which band applies, because you are judging
from the URL and title alone - not from the page content. Read its path:

  https://site.gov/                     bare domain, no path -> a site root. At most 0.6.
  https://site.gov/datasets             a listing of many datasets -> at most 0.6, it is
                                        an index, not a dataset
  https://site.gov/datasets/parcels-2024   a path segment naming one specific dataset
                                        -> this is the 0.9-1.0 shape

A good site does not lift its own front page into a high band: the front page of the
best cadastral portal in the country is still a front page, and scores lower than the
single dataset page it links to. Reserve 1.0 for an exact, named, directly downloadable
match with no reservations. If you would give every URL the same score, you are not
making the judgement: the whole point is to separate the exact matches from the merely
relevant.

Also name the organization or entity that PUBLISHES the data - the body behind the
resource, not the site hosting it. Two datasets on one national portal published by two
different agencies have different publishers. Be specific when you can, generic when 
you cannot. Use "unknown" if you have no idea.

Reply with JSON only, with the keys in exactly this order:
{"assessment": "<one short sentence: what the page actually is, and how well it fits>",
 "is_data_source": true|false, "publisher": "<publishing org>", "confidence": 0.0-1.0}"""


def _normalize(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _priority(cand: Candidate) -> tuple[int, int]:
    """Which duplicate to keep: already-triaged first, then catalog hits."""
    return (cand.confidence is not None, cand.source.startswith("catalog:"))


def dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse by normalized URL, preferring already-triaged hits, then cap per domain."""
    best: dict[str, Candidate] = {}
    for cand in candidates:
        key = _normalize(cand.url)
        existing = best.get(key)
        if existing is None or _priority(cand) > _priority(existing):
            best[key] = cand

    per_domain: dict[str, int] = {}
    kept = []
    for cand in best.values():
        domain = _domain(cand.url)
        if per_domain.get(domain, 0) >= config.MAX_PER_DOMAIN:
            print(
                f"    [dedupe] drop (domain cap {config.MAX_PER_DOMAIN} for {domain}) {cand.url}"
            )
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        kept.append(cand)
    return kept


def prefilter(candidates: list[Candidate]) -> list[Candidate]:
    """Drop non-data media by MIME family - no model calls, no maintained lists.

    mimetypes is the stdlib mapping, so new formats arrive with Python rather than
    with us. Data formats (.zip, .geojson, .csv) classify as data and survive - the
    Kenya run's top pick was a .zip, so getting that wrong would throw away the best
    candidate.

    Unknown types are kept on purpose: triage is the judge, this layer only skips
    the cases where a model call is obviously wasted. Everything the old domain
    blocklist covered (social, forums, wikis, search-result pages) is now named in
    TRIAGE_SYSTEM instead of matched against a host list nobody would maintain.
    """
    kept = []
    for cand in candidates:
        mime = mimetypes.guess_type(cand.url)[0] or ""
        if mime.split("/")[0] in ("image", "video", "audio") or mime == "application/pdf":
            print(f"    [prefilter] drop ({mime}) {cand.url}")
            continue
        kept.append(cand)
    return kept


def _report_spread(survivors: list[Candidate]) -> None:
    """Log how much the triage scores actually discriminate.

    The sort below is only meaningful if the scores differ. A model that answers 1.0
    for everything turns it into a no-op and the final cut silently becomes "first N
    found" instead of "best N" - which is exactly what llama3.1:8b used to do. Cheap
    to print, and it makes the regression visible in the run log instead of only in a
    diff of two output files.
    """
    scores = [c.confidence for c in survivors if c.confidence is not None]
    if not scores:
        return
    print(
        f"    [triage] confidence spread {min(scores):.2f}-{max(scores):.2f} "
        f"across {len(scores)} survivor(s)"
    )
    if len(set(scores)) == 1 and len(scores) > 1:
        print(
            f"    [triage] WARNING: all {len(scores)} survivors share confidence "
            f"{scores[0]:.2f} - ranking is insertion order, not quality"
        )


def triage(
    country: str, use_case: str, candidates: list[Candidate]
) -> tuple[list[Candidate], list[Candidate]]:
    """One cheap TRIAGE_MODEL call per candidate. Returns (survivors, unresolved).

    Three outcomes, not two. A call that comes back with no parseable verdict is NOT a
    negative verdict - coercing it into one silently deleted the best France candidate.
    Such candidates are excluded from the results (so junk can never ride in on a failed
    call) but handed back as `unresolved` so nothing vanishes without a trace.
    """
    survivors: list[Candidate] = []
    unresolved: list[Candidate] = []
    for cand in candidates:
        if cand.confidence is not None:
            # Carried over from an earlier iteration - it already passed triage, so
            # don't pay for the call again and don't risk a flaky re-judgement
            # dropping something we already accepted.
            print(f"    [triage] keep (carried from earlier iteration) {cand.url}")
            survivors.append(cand)
            continue

        data = chat_json(
            config.TRIAGE_MODEL,
            [
                {"role": "system", "content": TRIAGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Country: {country}\nUse case: {use_case}\n"
                        f"URL: {cand.url}\nTitle: {cand.title}\n"
                        f"What the finder said: {cand.rationale}"
                    ),
                },
            ],
        )

        if data is None or "is_data_source" not in data:
            # No answer - not a "no". Keep it out of the results, but hand it back.
            cand.rationale = (
                "triage returned no parseable verdict after "
                f"{config.CHAT_JSON_ATTEMPTS} attempts"
            )
            print(f"    [triage] unresolved (no verdict) {cand.url}")
            unresolved.append(cand)
            continue

        is_data = bool(data.get("is_data_source"))
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        publisher = str(data.get("publisher") or "").strip() or None
        if publisher and publisher.lower() == "unknown":
            publisher = None

        verdict = "keep" if is_data and confidence >= config.TRIAGE_CONFIDENCE_FLOOR else "drop"
        print(
            f"    [triage] {verdict} (data={is_data} conf={confidence:.2f} "
            f"publisher={publisher or '?'}) {cand.url}"
        )
        if verdict == "keep":
            cand.confidence = confidence
            cand.publisher = publisher
            # `assessment` is the current key; `rationale` is accepted so a model that
            # keeps emitting the old name still gives us a sentence.
            note = data.get("assessment") or data.get("rationale")
            if note:
                cand.rationale = str(note)
            survivors.append(cand)

    survivors.sort(key=lambda c: c.confidence or 0.0, reverse=True)
    _report_spread(survivors)
    return survivors[: config.MAX_FINAL_CANDIDATES], unresolved


def merge_and_triage(
    country: str, use_case: str, raw: list[Candidate]
) -> tuple[list[Candidate], list[Candidate]]:
    deduped = dedupe(raw)
    filtered = prefilter(deduped)
    print(f"  [merge] {len(raw)} raw -> {len(deduped)} deduped -> {len(filtered)} after prefilter")
    survivors, unresolved = triage(country, use_case, filtered)
    print(
        f"  [triage] {len(filtered)} -> {len(survivors)} survived, "
        f"{len(unresolved)} unresolved"
    )
    return survivors, unresolved
