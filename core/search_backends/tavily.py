"""Tavily-backed SearchBackend: Search + Extract APIs. Needs TAVILY_API_KEY.

Extract renders JS server-side, unlike a plain GET - see resource_links.py's module
docstring for the one place that deliberately keeps using a plain GET instead.
"""

import re
from urllib.parse import urljoin

import requests

from core import config
from core.search_backends.base import FetchedPage, Link, SearchBackend, SearchHit

# Markdown link syntax, e.g. [some text](path/or/https://example.com/path). The negative
# lookbehind skips image syntax (![alt](url)) - Extract returns markdown, not HTML, so this
# is how links are recovered since the API has no separate links field a raw <a href> scan
# would give. The href itself may be relative, so it's resolved (and filtered to http(s))
# with urljoin() at the call site, not restricted to absolute URLs here.
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]{0,80})\]\(([^\s)]+)\)")


class TavilySearchBackend(SearchBackend):
    name = "tavily"

    def __init__(self) -> None:
        if not config.TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set - add it to .env (see .env.example)")

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(
            f"https://api.tavily.com/{path}",
            headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            json=payload,
            timeout=config.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def web_search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        data = self._post(
            "search",
            {
                "query": query,
                "search_depth": config.TAVILY_SEARCH_DEPTH,
                "max_results": max_results,
            },
        )
        return [
            SearchHit(title=r.get("title", ""), url=r.get("url", ""), content=r.get("content", ""))
            for r in data.get("results", [])
            if isinstance(r, dict) and str(r.get("url", "")).startswith("http")
        ]

    def fetch_page(self, url: str, max_chars: int = 3000, max_links: int = 30) -> FetchedPage:
        data = self._post(
            "extract",
            {"urls": [url], "extract_depth": config.TAVILY_EXTRACT_DEPTH, "format": "markdown"},
        )
        results = data.get("results") or []
        if not results:
            failed = (data.get("failed_results") or [{}])[0]
            raise RuntimeError(failed.get("error", "Tavily could not extract this page"))
        raw = results[0].get("raw_content", "") or ""
        text = " ".join(raw.split())[:max_chars]

        links, seen = [], set()
        for match in _MD_LINK_RE.finditer(raw):
            absolute = urljoin(url, match.group(2))
            if not absolute.startswith("http") or absolute in seen:
                continue
            seen.add(absolute)
            links.append(Link(text=match.group(1).strip(), url=absolute))
            if len(links) >= max_links:
                break
        return FetchedPage(url=url, text=text, links=links)
