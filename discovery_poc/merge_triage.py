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

Answer true only if the page is (or directly serves) a dataset, download, API or data
catalog entry that plausibly covers the requested country and use case.
Answer false for blog posts, news, tutorials, product marketing, generic homepages,
social media, forums and Q&A threads, encyclopedia articles, search-engine result pages,
and for data that is about something else entirely.

Also name the organization or entity that PUBLISHES the data - the body behind the
resource, not the site hosting it. Two datasets on one national portal published by two
different agencies have different publishers. Be specific when you can ("INSEE", "IGN",
"OpenStreetMap contributors"), generic when you cannot ("commercial GIS vendor",
"data.gouv.fr open data platform"). Use "unknown" if you have no idea.

Reply with JSON only:
{"is_data_source": true|false, "confidence": 0.0-1.0, "publisher": "<publishing org>",
 "rationale": "<one short sentence>"}"""


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


def triage(country: str, use_case: str, candidates: list[Candidate]) -> list[Candidate]:
    """One cheap TRIAGE_MODEL call per candidate: real data endpoint vs. noise."""
    survivors = []
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
        ) or {}

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
            if data.get("rationale"):
                cand.rationale = str(data["rationale"])
            survivors.append(cand)

    survivors.sort(key=lambda c: c.confidence or 0.0, reverse=True)
    return survivors[: config.MAX_FINAL_CANDIDATES]


def merge_and_triage(country: str, use_case: str, raw: list[Candidate]) -> list[Candidate]:
    deduped = dedupe(raw)
    filtered = prefilter(deduped)
    print(f"  [merge] {len(raw)} raw -> {len(deduped)} deduped -> {len(filtered)} after prefilter")
    survivors = triage(country, use_case, filtered)
    print(f"  [triage] {len(filtered)} -> {len(survivors)} survived")
    return survivors
