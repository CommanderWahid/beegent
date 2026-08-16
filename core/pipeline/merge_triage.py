"""
Merge & triage: dedupe, drop obvious junk for free, then classify what survives.

Three layers, cheap first - dedupe and the deterministic pre-filter cost nothing and kill
the easy junk before any model call. No network I/O happens in this module at all:
resource_url and description are both filled in upstream, by search_explore, the moment a
candidate is found - this module only ever reads what it's handed.
"""

import mimetypes
from urllib.parse import urlparse

from core import config
from core.llm import chat_json
from core.schemas import Candidate

TRIAGE_SYSTEM = """You judge how good a usable geospatial DATA SOURCE a URL is for a
specific request - not merely a page that mentions the topic. Report a single number,
confidence, that answers both "is this actually a dataset, download, API or data catalog
entry" and "how DIRECTLY does it serve the requested country and use case".

You are given the URL, its title, a description of the page's content written by the
search step that found it, and whether a direct resource endpoint (an actual download
link or API) was identified for it. Use whatever evidence you have; more evidence should
sharpen the score, not just raise it. One rule overrides everything else below: no
identified resource endpoint means the ceiling is 0.4, not a target to round toward -
repeated as the last line of this prompt, right before you answer.

Below 0.5 is reserved for candidates with no identified resource endpoint:

  0.1-0.4  no resource endpoint identified. Could be a blog, news, tutorial, marketing
           page, generic homepage, forum/social/encyclopedia/search-result page, the
           wrong topic entirely, or even a real dataset's own landing page that nobody
           could confirm a download or API for - all of these land here, regardless of
           how well the title or description otherwise matches.

0.5 and above is reserved for candidates with a confirmed resource endpoint. Within that
range, the description is your best evidence: it can tell you the actual geographic
coverage, theme and vintage of the data, none of which the URL or title alone can
promise. Weigh it accordingly:

  0.5-0.6  a confirmed endpoint, but coverage is unclear or unverified - a catalog/portal
           entry that probably leads to the right data, or a description that does not
           confirm the country or use case
  0.6-0.7  a confirmed endpoint for a real dataset, but broader or adjacent - a national
           theme layer that contains what was asked for, a superset, or one region of the
           country, whether you read that from the description or infer it from the
           URL/title
  0.7-0.8  a confirmed endpoint whose description or URL/title makes a solid case for
           both the country and the use case, with a minor reservation - one of the two
           is implicit rather than stated outright, or the description is thin (title
           only, page never actually fetched)
  0.8-0.9  a confirmed endpoint whose description (or an unambiguous URL naming the exact
           dataset and its scope, if the description is thin) explicitly supports both
           the country and the use case, with only a small uncertainty left - format,
           freshness, exact boundary vintage
  0.9-1.0  a specific named dataset, download or API confirmed by its own description (or
           an exact, unambiguous URL) to match both the country and the use case, with no
           reservations at all

Before answering, check your own assessment for a reservation about scope, breadth or
authority - "broader", "adjacent", "supranational", "primarily about X rather than Y",
"community-sourced rather than official", and similar. A stated reservation like that
caps you at 0.6-0.7: it is a direct admission this is not the exact match the 0.7+ bands
require. A score of 0.8 or higher and a hedge in the same assessment is a contradiction -
if you write the hedge, use the lower band.

A good site does not lift its own front page into a high band: the front page of the best
cadastral portal in the country is still a front page, and scores lower than the single
dataset page it links to. Reserve 1.0 for an exact, named, directly downloadable match with
no reservations. If you would give every URL the same score, you are not making the
judgement: the whole point is to separate the exact matches from the merely relevant.

Also name the organization or entity that PUBLISHES the data - the body behind the
resource, not the site hosting it. Two datasets on one national portal published by two
different agencies have different publishers. Be specific when you can, generic when you
cannot. Use "unknown" if you have no idea.

Last check before you answer: if no resource endpoint was identified for this candidate,
confidence MUST be 0.4 or below - and 0.4 is the ceiling, not a safe middle value, so if
you are unsure, go lower (0.1-0.3) rather than rounding up to it. Nothing downstream
double-checks this number: you are the only place it is enforced. Being the right topic,
or even having a good description, is not the same as being data you can actually fetch.

Reply with JSON only, with the keys in exactly this order:
{"assessment": "<one short sentence: what the page actually is, and how well it fits>",
 "publisher": "<publishing org>", "confidence": 0.0-1.0}"""


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
    """
    Drop non-data media by MIME family - no model calls, no maintained lists.

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
    """
    Log how much the triage scores actually discriminate.

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
    """
    One cheap TRIAGE_MODEL call per candidate. Returns (survivors, unresolved).

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
                        f"Direct resource endpoint: {cand.resource_url or 'none identified'}\n"
                        f"Description: {cand.description or '(not available)'}\n"
                        f"What the finder said: {cand.rationale}"
                    ),
                },
            ],
        )

        if data is None or "confidence" not in data:
            # No answer - not a "no". Keep it out of the results, but hand it back.
            cand.rationale = (
                "triage returned no parseable verdict after "
                f"{config.CHAT_JSON_ATTEMPTS} attempts"
            )
            print(f"    [triage] unresolved (no verdict) {cand.url}")
            unresolved.append(cand)
            continue

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        publisher = str(data.get("publisher") or "").strip() or None
        if publisher and publisher.lower() == "unknown":
            publisher = None

        # No code-side cap here: TRIAGE_SYSTEM instructs the model to score at most 0.4
        # when cand.resource_url is None, and that instruction is the only enforcement -
        # trusted rather than re-applied. If the model ignores it, the escalation gate in
        # critic.py ("no candidate has a fetchable resource endpoint") is what catches it.
        verdict = "keep" if confidence >= config.TRIAGE_CONFIDENCE_FLOOR else "drop"
        print(
            f"    [triage] {verdict} (conf={confidence:.2f} publisher={publisher or '?'}) "
            f"{cand.url}"
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
