"""Find the endpoint that actually serves a page's data, using open standards only.

No hostnames, no per-portal URL templates, no country tables. A vendor list is always one
country behind the one someone asks for next, so this leans on things every well-behaved
portal already speaks:

  Content-Type   the server's own answer to "is this data or a web page". Authoritative and
                 universal - landing pages come back text/html, resources come back
                 application/zip, application/json, text/csv...
  DCAT/JSON-LD   schema.org Dataset and DCAT-AP distributions, published by open-data portals
                 worldwide (Google Dataset Search requires them, so portals maintain them).
  URL shape      data file extensions, and path conventions like OGC wfs/wms/ows or /download.
                 Conventions, not products.

Returning None is always allowed and always safe: triage caps a candidate with no resource
endpoint, so a wrong guess here would launder a landing page into a real candidate, while a
miss only costs one candidate its confidence.
"""

import json
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config

# Formats that are the data rather than a page about it.
_DATA_EXTENSIONS = (
    ".zip",
    ".gz",
    ".7z",
    ".geojson",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".gpkg",
    ".shp",
    ".kml",
    ".kmz",
    ".gml",
    ".xlsx",
    ".xls",
    ".tif",
    ".tiff",
    ".parquet",
    ".gdb",
    ".nc",
    ".grib",
)

# Path fragments that mark an endpoint rather than a document. The OGC services (wfs/wms/wcs/
# ows) are international standards; the rest are naming conventions that long predate any
# particular portal product.
_RESOURCE_PATH_HINTS = (
    "/api/",
    "/download",
    "/export",
    "/dump",
    "/wfs",
    "/wms",
    "/wcs",
    "/ows",
    "/rest/services",
    "/geoserver",
)

# Content types that mean "this response is a page", not "this response is data". Everything
# else is treated as data - an allowlist would have to guess at every format on earth, while
# the set of things browsers render as a document is small and stable.
_PAGE_CONTENT_TYPES = ("text/html", "application/xhtml+xml")

# Types that describe a dataset rather than being one. Portals advertise these as alternate
# representations of a dataset page (RDF/N3/Turtle for DCAT, RSS/Atom for change feeds), and
# they are not html, so a bare "is it html?" test happily accepts a metadata document as the
# data. They are worth mining for the links they contain, never worth returning.
_METADATA_CONTENT_TYPES = (
    "application/rdf+xml",
    "application/ld+json",
    "application/n-triples",
    "application/trig",
    "text/n3",
    "text/turtle",
    "application/rss+xml",
    "application/atom+xml",
)

# JSON keys that carry a distribution's URL in schema.org and DCAT. Compared casefolded, so
# downloadUrl / downloadURL / download_url all match.
_DISTRIBUTION_URL_KEYS = ("downloadurl", "contenturl", "accessurl", "url")

# For pulling links out of a metadata document without a per-format parser.
_URL_RE = re.compile(r"https?://[^\s\"'<>\\]+")

# One memo per process. The same URL turns up across angles and is carried between iterations,
# and probing is the expensive part of this module.
_resolved: dict[str, str | None] = {}
_probes = 0


def _budget_left() -> bool:
    return _probes < config.MAX_RESOURCE_PROBES


def looks_like_resource(url: str) -> bool:
    """True when a URL's own shape says it serves data. No network."""
    path = urlparse(url).path.lower()
    return path.endswith(_DATA_EXTENSIONS) or any(h in path for h in _RESOURCE_PATH_HINTS)


def _rank(url: str) -> int:
    """Strength of the evidence that a URL is the data itself. Lower is better.

    A data file extension is unambiguous. A path hint like /api/ or /download is not: portals
    hang their metadata endpoints off exactly those paths (`/download_metadata?format=json`,
    `/api/3/action/package_show`), and those answer with a perfectly respectable
    application/json that no content-type check can tell apart from real data. So extensions
    sort first, and anything self-describing as metadata sorts last.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()
    if "metadata" in path or "metadata" in parsed.query.lower():
        return 2
    if path.endswith(_DATA_EXTENSIONS):
        return 0
    return 1


def _by_evidence(urls: list[str]) -> list[str]:
    """Stable sort by evidence strength; ties keep discovery order, so a publisher's own
    JSON-LD declaration still outranks a link scraped from the page body."""
    return sorted(urls, key=_rank)


def serves_data(url: str) -> bool:
    """Ask the server whether this URL returns data rather than a web page.

    A ranged GET rather than a HEAD: plenty of servers answer HEAD with 405 or lie about the
    content type, and a couple of KB is cheap enough to just look. Redirects are followed, so
    the content type is the one at the end of the chain.
    """
    global _probes
    if not url.startswith("http") or not _budget_left():
        return False
    _probes += 1
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": config.USER_AGENT,
                "Range": f"bytes=0-{config.RESOURCE_PROBE_BYTES}",
            },
            timeout=config.HTTP_TIMEOUT,
            stream=True,
        )
    except Exception:
        return False
    with resp:
        if resp.status_code >= 400:
            return False
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        # No content type at all is not evidence of data; fall back to the URL's shape.
        if not content_type:
            return looks_like_resource(url)
        return content_type not in _PAGE_CONTENT_TYPES + _METADATA_CONTENT_TYPES


# schema.org types that commonly carry their own `url` field but are never themselves a
# distribution - harvesting bare "url" from these turns an org's contact page or a person's
# homepage into a fake dataset resource. downloadUrl/contentUrl/accessUrl are unambiguous
# distribution vocabulary and are never filtered by this.
_NON_DISTRIBUTION_TYPES = {
    "contactpoint", "organization", "person", "postaladdress",
    "website", "webpage", "imageobject", "place",
}


def _walk_json_for_urls(node, found: list[str]) -> None:
    """Collect distribution URLs from a parsed schema.org/DCAT blob, any nesting."""
    if isinstance(node, dict):
        node_type = str(node.get("@type", "")).casefold()
        for key, value in node.items():
            if key.casefold() in _DISTRIBUTION_URL_KEYS and isinstance(value, str):
                bare_url_on_non_distribution = (
                    key.casefold() == "url" and node_type in _NON_DISTRIBUTION_TYPES
                )
                if value.startswith("http") and not bare_url_on_non_distribution:
                    found.append(value)
            else:
                _walk_json_for_urls(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_json_for_urls(item, found)


def resource_links_from_html(html: str, base_url: str) -> list[str]:
    """Candidate resource URLs declared on a page, best evidence first.

    A publisher's own structured declaration of where its data lives beats guessing from link
    shapes, so JSON-LD comes first. Note what is NOT here: `<link rel="alternate">`. Portals
    use it to advertise metadata serializations of the dataset page - RDF, N3, RSS - which are
    perfectly fetchable and perfectly useless as data. Those are handled by
    metadata_documents_from_html() below, as things to read rather than things to return.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        absolute = urljoin(base_url, raw)
        if absolute.startswith("http") and absolute not in out and absolute != base_url:
            out.append(absolute)

    # 1. schema.org Dataset / DCAT distributions embedded in the page.
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            blob = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        found: list[str] = []
        _walk_json_for_urls(blob, found)
        for url in found:
            add(url)

    # 2. Ordinary links whose shape says "data".
    for anchor in soup.find_all("a", href=True):
        if looks_like_resource(urljoin(base_url, anchor["href"])):
            add(anchor["href"])

    return _by_evidence(out)


def metadata_documents_from_html(html: str, base_url: str) -> list[str]:
    """Alternate representations of the page that are metadata worth mining for links."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for link in soup.find_all("link", rel=True, href=True):
        if "alternate" not in [r.lower() for r in link.get("rel", [])]:
            continue
        declared = (link.get("type") or "").split(";")[0].strip().lower()
        if declared in _METADATA_CONTENT_TYPES:
            absolute = urljoin(base_url, link["href"])
            if absolute.startswith("http") and absolute not in out:
                out.append(absolute)
    return out


def _links_in_metadata(url: str) -> list[str]:
    """Resource-shaped URLs mentioned anywhere in a metadata document.

    One regex covers JSON-LD, RDF/XML, N3 and Turtle alike - no per-format parser, and no risk
    from a sloppy extraction, because every hit still has to pass serves_data() before it is
    returned to anyone.
    """
    global _probes
    if not _budget_left():
        return []
    _probes += 1
    try:
        resp = requests.get(
            url, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT
        )
        resp.raise_for_status()
    except Exception:
        return []
    out: list[str] = []
    for found in _URL_RE.findall(resp.text):
        cleaned = found.rstrip('">,)\\')
        if looks_like_resource(cleaned) and cleaned not in out:
            out.append(cleaned)
    return _by_evidence(out)


def _fetch_html(url: str) -> str | None:
    global _probes
    if not _budget_left():
        return None
    _probes += 1
    try:
        resp = requests.get(
            url, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT
        )
        resp.raise_for_status()
    except Exception:
        return None
    return resp.text


def _first_confirmed(candidates: list[str]) -> str | None:
    """First candidate the server confirms actually serves data, within budget."""
    for candidate in candidates[: config.MAX_RESOURCE_CANDIDATES_PER_PAGE]:
        if not _budget_left():
            return None
        if serves_data(candidate):
            return candidate
    return None


def resolve_resource_url(url: str) -> str | None:
    """The endpoint serving this page's data, or None if none can be confirmed.

    Never returns an unverified URL: whatever comes back has been probed and answered with a
    non-page content type.
    """
    if not url.startswith("http"):
        return None
    if url in _resolved:
        return _resolved[url]

    resolved: str | None = None
    if serves_data(url):
        resolved = url  # the page we were given is itself the resource
    else:
        html = _fetch_html(url)
        if html:
            resolved = _first_confirmed(resource_links_from_html(html, url))
            if resolved is None:
                # Nothing linked directly. The page may still describe its distributions in a
                # DCAT/RDF document it advertises - common where the download links themselves
                # are rendered by JavaScript and so invisible to us.
                for doc in metadata_documents_from_html(html, url)[:2]:
                    resolved = _first_confirmed(_links_in_metadata(doc))
                    if resolved:
                        break

    _resolved[url] = resolved
    return resolved


def probe_count() -> int:
    """Requests spent resolving so far - for run logging."""
    return _probes


def budget_exhausted() -> bool:
    """True once probing has stopped. Callers should say so out loud: past this point every
    unresolved candidate looks unfetchable, and triage will cap and drop it on that basis."""
    return not _budget_left()
