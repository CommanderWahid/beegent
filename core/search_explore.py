"""Search & explore worker: one LLM agent per search angle, with a hard tool budget.

The agent searches once, and only fetches a page when a result looks like a
homepage/landing page rather than the actual data resource - then follows one or
two links deeper to find the real data page.

Known limitations (accepted for this POC):
  - web_search scrapes DuckDuckGo Lite. Keyless and free, but unofficial: rate
    limits or markup changes will break it. Swap point is web_search() alone.
  - fetch_page does a plain HTTP GET with no JS rendering, so JS-only data
    portals look empty.
"""

import json
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import config
from llm import chat_json, chat_tools, to_message_dict
from resource_links import looks_like_resource
from schemas import Candidate, SearchAngle

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

Reply with JSON only:
{"candidates": [{"url": "...", "title": "...",
                 "resource_url": "<direct download or API endpoint, or null>",
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


def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Keyless web search via the DuckDuckGo Lite endpoint."""
    resp = requests.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results, seen = [], set()
    for a in soup.select("a.result-link"):
        url = a.get("href", "")
        if not url.startswith("http") or "duckduckgo.com" in url or url in seen:
            continue
        seen.add(url)
        results.append({"title": a.get_text(strip=True), "url": url})
        if len(results) >= max_results:
            break
    return results


def fetch_page(url: str, max_chars: int = 3000, max_links: int = 30) -> dict:
    """Plain GET + tag-stripped text + outbound links. No JS rendering."""
    resp = requests.get(
        url, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())[:max_chars]

    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, a["href"])
        if not absolute.startswith("http") or absolute in seen:
            continue
        seen.add(absolute)
        links.append({"text": a.get_text(strip=True)[:80], "url": absolute})
        if len(links) >= max_links:
            break
    return {"url": url, "text": text, "links": links}


def _reported_resource_url(raw: dict, url: str, seen: dict[str, int]) -> str | None:
    """The endpoint the model reported, if it is allowed to count. No network.

    Held to the same standard as `url`: it must have actually turned up in a tool result
    (`seen` holds every search hit and every link off a fetched page). A model that invents a
    plausible-looking /api/ path would otherwise launder a landing page into a high-confidence
    candidate, which is the exact failure this field exists to catch.

    Only the cheap, local half of the job lives here, because `seen` exists nowhere else.
    Candidates left at None are picked up by merge_triage.resolve_resources(), which probes.
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

    searches = fetches = 0
    hops: dict[str, int] = {}  # url -> page-fetches it took to surface it
    titles: dict[str, str] = {}
    search_hits: list[dict] = []

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
                        hits = web_search(args.get("query", angle.description))
                    except Exception as exc:
                        hits = []
                        print(f"    [web_search] failed: {exc}")
                    print(f"    [web_search] {args.get('query', '')!r} -> {len(hits)} results")
                    for hit in hits:
                        hops.setdefault(hit["url"], 0)
                        titles.setdefault(hit["url"], hit["title"])
                    search_hits.extend(hits)
                    result = hits

            elif name == "fetch_page":
                url = args.get("url", "")
                if fetches >= config.MAX_FETCHES_PER_ANGLE:
                    result = "budget exhausted: no fetches left, answer with what you have"
                elif not url.startswith("http"):
                    result = f"invalid url: {url!r}"
                else:
                    fetches += 1
                    try:
                        page = fetch_page(url)
                        parent_hops = hops.get(url, 0)
                        for link in page["links"]:
                            hops.setdefault(link["url"], parent_hops + 1)
                            titles.setdefault(link["url"], link["text"])
                        result = page
                        print(f"    [fetch_page] {url} ({len(page['text'])} chars)")
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
                resource_url=_reported_resource_url(raw, url, hops),
            )
        )

    if not candidates and search_hits:
        # POC fallback: the model gave us nothing parseable, so pass the raw search
        # hits downstream and let triage do the judging.
        print("    [explore] no candidates from model, falling back to raw search hits")
        candidates = [
            Candidate(
                url=hit["url"],
                title=hit["title"],
                source="search_explore",
                hops=0,
                rationale="raw search hit (model returned no structured candidates)",
                # No model verdict to work from; merge_triage.resolve_resources() will probe
                # these like any other candidate.
            )
            for hit in search_hits[:5]
        ]
    return candidates
