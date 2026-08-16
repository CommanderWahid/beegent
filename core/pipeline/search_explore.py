"""Search & explore worker: one LLM agent per search angle, with a hard tool budget.

The agent searches once, and only fetches a page when a result looks like a
homepage/landing page rather than the actual data resource - then follows one or
two links deeper to find the real data page.

Owns a candidate's full factual record, not just discovery: resource_url extraction
(_resolve_resource_urls(), via resource_links.py) and a written description of each
page's content both happen here, the moment a candidate is found - not later in
merge_triage, which only dedupes, prefilters and scores what this module hands it.

Search and page-fetching go through a pluggable SearchBackend (see
core/search_backends/) - which provider is active is config.SEARCH_BACKEND, not something
this module hardcodes. Today's only implementation (core/search_backends/tavily.py) is a
real authenticated JSON API with no scraping fragility and no anti-bot blocking, and its
Extract call renders JS server-side, so JS-only portals are no longer an automatic dead
end for this step specifically (resource_links.py's own resolve_resource_url() still
can't render JS - see CLAUDE.md). Cost is bounded the same way regardless of backend:
MAX_SEARCHES_PER_ANGLE and MAX_FETCHES_PER_ANGLE cap it per angle.
"""

import json
from dataclasses import asdict

from core import config
from core.llm import chat_json, chat_tools, to_message_dict
from core.pipeline.resource_links import (
    budget_exhausted,
    looks_like_resource,
    probe_count,
    resolve_resource_url,
    serves_data,
)
from core.schemas import Candidate, SearchAngle
from core.search_backends import SearchHit, get_search_backend

SYSTEM = """You find real, downloadable geospatial data sources on the internet.

You have two tools:
  web_search(query)  - use ONCE, with a precise query for the angle you were given.
  fetch_page(url)    - use ONLY when a search result looks like a homepage, portal
                       landing page or category page rather than an actual dataset.
                       The page text and its links come back; follow at most one or
                       two links deeper to reach the real data page.

You may fetch at most {max_fetches} pages in total. Do not fetch a URL that already
looks like a dataset/resource/download page - that is already what we want.

Whenever you fetch a page, look in its links for the endpoint that actually serves the
data: an export or download button, a direct file link (.geojson, .zip, .csv, .gpkg), or
an API link (WFS/WMS, /api/, /rest/services). Note it - a page describing a dataset is
worth much less than the link that hands you the dataset.
When you have what you need, stop calling tools and say so."""

FINAL = """List the data sources you found for this angle.

Only include URLs you actually saw in tool results. Prefer dataset/download/API pages
over homepages. Include nothing you are not reasonably confident is a data source.
It is fine to return an empty list.

For each one, set resource_url to the link that actually serves the data - a direct file
download or an API endpoint you saw in a tool result. If you did not see one, set it to
null. Do not repeat the page url there and do not construct a plausible-looking endpoint:
null is the correct, useful answer when no download or API link was found.

Also write a description: 1-2 sentences on what the page's content actually shows -
geographic coverage, theme, format, vintage - based on what you saw in a fetch_page
result. If you never fetched this page (found only via web_search, no tool result to
read), say so plainly ("title/URL only, page not fetched") rather than guessing content
you never saw - a downstream step relies on this being honest, not persuasive.

Reply with JSON only:
{"candidates": [{"url": "...", "title": "...",
                 "resource_url": "<direct download or API endpoint, or null>",
                 "description": "<what the page's content shows, or 'title/URL only, page not fetched'>",
                 "rationale": "<one sentence: what data is there>"}]}"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return result titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page and return its text and outbound links.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def _reported_resource_url(raw: dict, url: str, seen: dict[str, int]) -> str | None:
    """The endpoint the model reported, if it is allowed to count. No network.

    Held to the same standard as `url`: it must have actually turned up in a tool result
    (`seen` holds every search hit and every link off a fetched page). A model that invents a
    plausible-looking /api/ path would otherwise launder a landing page into a high-confidence
    candidate, which is the exact failure this field exists to catch.

    Only the cheap, local half of the job lives here, because `seen` exists nowhere else.
    Candidates left at None are picked up by _resolve_resource_urls() below, which probes.
    """
    reported = raw.get("resource_url")
    if not isinstance(reported, str) or not reported.startswith("http"):
        return None
    # Echoing the page back is only meaningful if the page is itself an endpoint.
    if reported == url and not looks_like_resource(url):
        return None
    if reported not in seen:
        print(f"    [resource] discarding unseen resource_url from model: {reported}")
        return None
    return reported


def _resolve_resource_urls(candidates: list[Candidate]) -> None:
    """
    Fill in (and sanity-check) each candidate's resource_url, in place.

    Moved here from merge_triage so a candidate's resource_url is settled the moment it's
    found, before dedupe/prefilter/triage ever see it - triage becomes a pure decision
    layer over what this step already gathered. resolve_resource_url() memoizes by URL
    for the whole run, so a candidate re-found by a later angle costs nothing to recheck.

    Already-set values get verified rather than trusted: a link really seen on a real page
    can still be dead, and a URL that 404s is not a resource.
    """
    for cand in candidates:
        if cand.resource_url:
            if serves_data(cand.resource_url):
                continue
            print(f"    [resource] reported endpoint does not serve data: {cand.resource_url}")
            cand.resource_url = None
        cand.resource_url = resolve_resource_url(cand.url)
        if cand.resource_url:
            print(f"    [resource] {cand.url}\n               -> {cand.resource_url}")

    found = sum(1 for c in candidates if c.resource_url)
    print(f"    [resource] {found}/{len(candidates)} resolved ({probe_count()} probes spent)")
    if budget_exhausted():
        print(
            f"    [resource] WARNING: probe budget ({config.MAX_RESOURCE_PROBES}) exhausted - "
            "candidates after this point look unfetchable and will be capped on that basis"
        )


def explore_angle(country: str, use_case: str, angle: SearchAngle) -> list[Candidate]:
    """Run one bounded search+explore loop for a single angle."""
    messages = [
        {"role": "system", "content": SYSTEM.format(max_fetches=config.MAX_FETCHES_PER_ANGLE)},
        {
            "role": "user",
            "content": (
                f"Country: {country}\nUse case: {use_case}\n"
                f"Search angle: {angle.description}\nWhy this angle: {angle.rationale}"
            ),
        },
    ]

    backend = get_search_backend()
    searches = fetches = 0
    hops: dict[str, int] = {}  # url -> page-fetches it took to surface it
    titles: dict[str, str] = {}
    snippets: dict[str, str] = {}  # url -> the backend's search content snippet
    search_hits: list[SearchHit] = []  # for the no-structured-candidates fallback below

    for _ in range(config.MAX_TOOL_TURNS_PER_ANGLE):
        msg = chat_tools(config.SEARCH_EXPLORE_MODEL, messages, TOOLS)
        if not msg.tool_calls:
            break
        messages.append(to_message_dict(msg))

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result: dict | list | str

            if name == "web_search":
                if searches >= config.MAX_SEARCHES_PER_ANGLE:
                    result = "budget exhausted: no searches left, use what you have"
                else:
                    searches += 1
                    try:
                        hits = backend.web_search(args.get("query", angle.description))
                    except Exception as exc:
                        hits = []
                        print(f"    [web_search] failed: {exc}")
                    print(f"    [web_search] {args.get('query', '')!r} -> {len(hits)} results")
                    for hit in hits:
                        hops.setdefault(hit.url, 0)
                        titles.setdefault(hit.url, hit.title)
                        snippets.setdefault(hit.url, hit.content)
                    search_hits.extend(hits)
                    result = [asdict(h) for h in hits]

            elif name == "fetch_page":
                url = args.get("url", "")
                if fetches >= config.MAX_FETCHES_PER_ANGLE:
                    result = "budget exhausted: no fetches left, answer with what you have"
                elif not url.startswith("http"):
                    result = f"invalid url: {url!r}"
                else:
                    fetches += 1
                    try:
                        page = backend.fetch_page(url)
                        parent_hops = hops.get(url, 0)
                        for link in page.links:
                            hops.setdefault(link.url, parent_hops + 1)
                            titles.setdefault(link.url, link.text)
                        result = asdict(page)
                        print(f"    [fetch_page] {url} ({len(page.text)} chars)")
                    except Exception as exc:
                        result = f"fetch failed: {exc}"
                        print(f"    [fetch_page] {url} failed: {exc}")
            else:
                result = f"unknown tool {name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)[:6000],
                }
            )

        if searches >= config.MAX_SEARCHES_PER_ANGLE and fetches >= config.MAX_FETCHES_PER_ANGLE:
            break

    messages.append({"role": "user", "content": FINAL})
    data = chat_json(config.SEARCH_EXPLORE_MODEL, messages)

    candidates: list[Candidate] = []
    for raw in (data or {}).get("candidates", []):
        url = raw.get("url", "") if isinstance(raw, dict) else ""
        if not url.startswith("http"):
            continue
        candidates.append(
            Candidate(
                url=url,
                title=raw.get("title") or titles.get(url, url),
                source="search_explore",
                hops=hops.get(url, 0),
                rationale=raw.get("rationale", ""),
                description=str(raw.get("description") or snippets.get(url, "")),
                resource_url=_reported_resource_url(raw, url, hops),
            )
        )

    if not candidates and search_hits:
        # POC fallback: the model gave us nothing parseable, so pass the raw search
        # hits downstream and let triage do the judging.
        print("    [explore] no candidates from model, falling back to raw search hits")
        candidates = [
            Candidate(
                url=hit.url,
                title=hit.title,
                source="search_explore",
                hops=0,
                rationale="raw search hit (model returned no structured candidates)",
                description=hit.content,
                # No model verdict to work from; _resolve_resource_urls() below will probe
                # these like any other candidate.
            )
            for hit in search_hits[:5]
        ]

    _resolve_resource_urls(candidates)
    return candidates
