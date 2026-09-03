"""WebTools - the deterministic web layer the geofetch agent drives."""

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import quote_plus, unquote, urljoin

from beegent import config

MAGIC_SIGNATURES = [
    (b"PAR1", "parquet"),
    (b"PK\x03\x04", "zip"),
    (b"\x1f\x8b", "gzip"),
    (b"SQLite format 3\x00", "sqlite/geopackage"),
    (b"%PDF", "pdf"),
    (b"7z\xbc\xaf", "7z"),
    (b"Rar!", "rar"),
    (b"\x89PNG", "png"),
    (b"II*\x00", "tiff/geotiff"),
    (b"MM\x00*", "tiff/geotiff"),
]

_URL_RE = re.compile(r"https?://[^\s\"'<>\\)\]]+")

# Esri names its geometries differently; map to the OGC names the rest of the world uses.
_ESRI_GEOMETRY = {
    "esriGeometryPoint": "Point",
    "esriGeometryMultipoint": "MultiPoint",
    "esriGeometryPolyline": "LineString",
    "esriGeometryPolygon": "Polygon",
    "esriGeometryEnvelope": "Polygon",
}

_FEATURE_RE = re.compile(r'"type"\s*:\s*"Feature"')
_GEOM_TYPE_RE = re.compile(r'"geometry"\s*:\s*\{\s*"type"\s*:\s*"([A-Za-z]+)"')
_FC_MARKER_RE = re.compile(r'"type"\s*:\s*"FeatureCollection"')
_XML_MEMBER_RE = re.compile(r"<(?:\w+:)?(?:featureMember|member)\b")
_XML_MATCHED_RE = re.compile(r'number(?:Matched|OfFeatures)\s*=\s*"(\d+)"')


@dataclass
class HttpResult:
    status: int
    headers: dict  # lower-cased keys
    body: bytes
    final_url: str
    error: str = ""


def normalize_url(url: str) -> str:
    """Comparison form: scheme+host+path, no query/fragment, no trailing slash."""
    return url.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def classify_magic(first_bytes: bytes) -> str:
    """Identify a payload from its leading bytes - the whole basis of verification."""
    for sig, name in MAGIC_SIGNATURES:
        if first_bytes.startswith(sig):
            return name
    head = first_bytes.lstrip()[:1]
    if head in (b"<",):
        return "xml/html-text"
    if head in (b"{", b"["):
        return "json-text"
    return "unknown"


def _shape(kind, count, exact, geometry="", paged=False):
    # paged: the reply carries service metadata, which is what separates an API from a file.
    return {"shape": kind, "feature_count": count, "count_is_exact": exact,
            "geometry_type": geometry, "paged": paged}


def _json_shape(doc: dict) -> dict | None:
    """A fully parsed JSON document - the only path that can report an exact count."""
    if "error" in doc:  # ArcGIS serves these with HTTP 200
        return None
    capped = bool(doc.get("exceededTransferLimit"))
    if doc.get("type") == "FeatureCollection":
        feats = doc.get("features") or []
        matched = doc.get("numberMatched")
        exact = isinstance(matched, int) and matched >= 0 and not capped
        geometry = (feats[0].get("geometry") or {}).get("type", "") if feats else ""
        served = any(k in doc for k in ("numberMatched", "numberReturned", "links"))
        return _shape("geojson_featurecollection", matched if exact else len(feats),
                      exact, geometry, served or capped)
    if "geometryType" in doc and "features" in doc:
        feats = doc.get("features") or []
        total = doc.get("count")
        exact = isinstance(total, int) and not capped
        return _shape("esrijson_featureset", total if exact else len(feats), exact,
                      _ESRI_GEOMETRY.get(doc.get("geometryType", ""), ""), True)
    return None


#: Service metadata that can appear BEFORE the features array, so a cut body still shows it.
_SERVICE_KEYS = ('"links"', "numberMatched", "numberReturned", "exceededTransferLimit")


def _text_shape(text: str) -> dict | None:
    """A truncated or XML body - features are countable, a total usually is not."""
    if _FC_MARKER_RE.search(text):
        geometry = _GEOM_TYPE_RE.search(text)
        # Cut mid-document, so the count is a lower bound and the tail metadata is gone.
        # "links" is the one service marker OGC API-Features puts at the TOP of the document.
        return _shape("geojson_featurecollection", len(_FEATURE_RE.findall(text)), False,
                      geometry.group(1) if geometry else "",
                      any(k in text for k in _SERVICE_KEYS))
    if "WFS_Capabilities" in text:  # metadata, not data - 0 features is the honest answer
        return _shape("wfs_capabilities", 0, True, paged=True)
    if re.search(r"<(?:\w+:)?FeatureCollection\b", text):
        matched = _XML_MATCHED_RE.search(text)
        return _shape("wfs_featurecollection",
                      int(matched.group(1)) if matched else len(_XML_MEMBER_RE.findall(text)),
                      bool(matched), paged=True)
    return None


def classify_api_shape(body: bytes, content_type: str = "") -> dict | None:
    """Identify a feature service from the STRUCTURE of its reply, never from its host."""
    if "html" in content_type.lower():
        return None
    text = body.decode("utf-8", "replace").lstrip()
    if text[:1] in ("{", "["):
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            return _text_shape(text)  # cut mid-document: count what is visible
        return _json_shape(doc) if isinstance(doc, dict) else None
    if text[:1] == "<":
        return _text_shape(text)
    return None


def default_transport(method: str, url: str, headers: dict, max_bytes: int) -> HttpResult:
    """Real HTTP transport (requests)."""
    import requests

    try:
        resp = requests.request(
            method, url, headers=headers, stream=True, timeout=config.HTTP_TIMEOUT,
            allow_redirects=True,
        )
        body = b""
        for chunk in resp.iter_content(chunk_size=65536):
            body += chunk
            if len(body) >= max_bytes:
                break
        resp.close()
        return HttpResult(
            resp.status_code,
            {k.lower(): v for k, v in resp.headers.items()},
            body[:max_bytes],
            resp.url,
        )
    except requests.RequestException as exc:
        return HttpResult(0, {}, b"", url, error=f"{type(exc).__name__}: {exc}")


# --- HTML -> text + links ---


class _LinkTextParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.links: list[tuple[str, str]] = []  # (text, absolute href)
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        d = dict(attrs)
        if tag == "a" and d.get("href"):
            self._href, self._anchor_text = d["href"], []
        if tag == "link" and d.get("href"):  # <link rel=alternate/atom/dcat href=...>
            rel = d.get("rel", "")
            self.links.append(
                (f"<link rel={rel} type={d.get('type', '')}>", urljoin(self.base, d["href"]))
            )
        if tag == "script" and d.get("src"):  # JS bundles hide API base URLs as literals
            self.links.append(("<script src>", urljoin(self.base, d["src"])))

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "a" and self._href:
            self.links.append(
                (" ".join(self._anchor_text).strip()[:120], urljoin(self.base, self._href))
            )
            self._href = None

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._href is not None:
            self._anchor_text.append(data.strip())
        if data.strip():
            self.text_parts.append(data.strip())


def summarize_html(body: bytes, base_url: str) -> dict:
    parser = _LinkTextParser(base_url)
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
    except Exception:
        pass
    text = re.sub(r"\s+", " ", " ".join(parser.text_parts))[: config.MAX_TEXT_CHARS]
    seen, links = set(), []
    for label, href in parser.links:
        if href not in seen:
            seen.add(href)
            links.append({"text": label, "href": href})
        if len(links) >= config.MAX_LINKS:
            break
    js_shell = len(text) < 300 and len(body) > 2000
    return {"text": text, "links": links, "looks_like_js_app_shell": js_shell}


# --- the tools ---


class WebTools:
    def __init__(
        self,
        transport: Callable = default_transport,
        log: Callable[[str], None] = lambda m: None,
    ):
        self.transport = transport
        self.max_requests = config.GEOFETCH_MAX_HTTP_REQUESTS
        self.requests_made = 0
        self.log = log

    def _guard(self):
        self.requests_made += 1
        if self.requests_made > self.max_requests:
            raise RuntimeError("HTTP request budget exhausted")

    def fetch_page(self, url: str, accept: str = "") -> dict:
        """Fetch a URL as text+links for HTML, or a truncated raw body for XML/JSON."""
        self._guard()
        self.log(f"  fetch_page {url}" + (f" (Accept: {accept})" if accept else ""))
        headers = {"User-Agent": "beegent-geofetch/1.0"}
        if accept:
            headers["Accept"] = accept
        r = self.transport("GET", url, headers, config.MAX_BODY_BYTES)
        if r.error:
            return {"url": url, "error": r.error}
        ctype = r.headers.get("content-type", "")
        out = {
            "requested_url": url,
            "final_url": r.final_url,
            "status": r.status,
            "content_type": ctype,
            "truncated": len(r.body) >= config.MAX_BODY_BYTES,
        }
        if r.final_url != url:
            out["note"] = "redirected - the final_url may reveal the real service"
        is_html = "html" in ctype or r.body[:200].lstrip().lower().startswith(
            (b"<!doctype html", b"<html")
        )
        if is_html:
            out.update(summarize_html(r.body, r.final_url))
        else:
            out["body"] = r.body[: config.MAX_BODY_BYTES].decode("utf-8", errors="replace")
        # Absolute URLs the anchors do NOT already carry - prose, XML metadata, JS bundles.
        seen = {link["href"] for link in out.get("links", [])}
        seen.add(url)
        urls = []
        for m in _URL_RE.finditer(r.body.decode("utf-8", errors="replace")):
            u = m.group(0).rstrip(".,;\"')\\")
            if u not in seen:
                seen.add(u)
                urls.append(u[:300])
            if len(urls) >= config.MAX_URLS_FOUND:
                break
        out["urls_found"] = urls
        return out

    # --- search ------------------------------------------------------------ #

    SEARCH_ENGINES = [
        ("ddg-html", "https://html.duckduckgo.com/html/?q={q}"),
        ("ddg-lite", "https://lite.duckduckgo.com/lite/?q={q}"),
        ("bing", "https://www.bing.com/search?q={q}"),
    ]
    ENGINE_DOMAINS = ("duckduckgo.com", "bing.com", "microsoft.com", "microsofttranslator.com")
    BLOCK_MARKERS = ("anomaly", "captcha", "challenge", "unusual traffic", "are you a robot")

    @staticmethod
    def _decode_result_href(href: str) -> str:
        """Unwrap engine redirect links (DDG uddg=, Bing ck/a u=a1<b64>)."""
        m = re.search(r"[?&]uddg=([^&]+)", href)
        if m:
            return unquote(m.group(1))
        if "/ck/a" in href:
            m = re.search(r"[?&]u=a1([^&]+)", href)
            if m:
                try:
                    import base64

                    pad = m.group(1) + "=" * (-len(m.group(1)) % 4)
                    return base64.urlsafe_b64decode(pad).decode("utf-8", "replace")
                except Exception:
                    return href
        return href

    def web_search(self, query: str) -> dict:
        """Keyless web search over a DuckDuckGo -> DDG-lite -> Bing fallback chain."""
        self.log(f"  web_search {query!r}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept-Language": "en;q=0.9,fr;q=0.8",
        }
        errors = []
        for engine, template in self.SEARCH_ENGINES:
            self._guard()
            url = template.format(q=quote_plus(query))
            r = self.transport("GET", url, headers, config.MAX_BODY_BYTES)
            if r.error or r.status != 200:
                errors.append(f"{engine}: {r.error or f'HTTP {r.status}'}")
                continue
            body_low = r.body[:4000].decode("utf-8", "replace").lower()
            results, seen = [], set()
            for link in summarize_html(r.body, url)["links"]:
                href = self._decode_result_href(link["href"])
                if (
                    href.startswith("http")
                    and href not in seen
                    and link["text"]
                    and not any(d in href for d in self.ENGINE_DOMAINS)
                ):
                    seen.add(href)
                    results.append({"title": link["text"], "url": href})
                if len(results) >= 10:
                    break
            if results:
                return {"query": query, "engine": engine, "results": results}
            blocked = any(mk in body_low for mk in self.BLOCK_MARKERS)
            errors.append(f"{engine}: " + ("bot-challenge page" if blocked else "0 links"))
        return {
            "query": query,
            "results": [],
            "engines_tried": errors,
            "note": "All engines returned nothing - this usually means bot-blocking, NOT "
            "that the dataset has no download. Rephrase (use the dataset's human name and "
            "local-language words like 'telechargement'/'descarga'/'download'), or continue "
            "with non-search techniques.",
        }

    # --- probe ------------------------------------------------------------- #

    def probe_url(self, url: str) -> dict:
        """Range-request the first bytes of a URL: status, size, payload type."""
        self._guard()
        self.log(f"  probe_url {url}")
        headers = {
            "User-Agent": "beegent-geofetch/1.0",
            "Range": f"bytes=0-{config.PROBE_BYTES - 1}",
        }
        r = self.transport("GET", url, headers, config.PROBE_BYTES)
        if r.error:
            return {"url": url, "ok": False, "error": r.error}
        total = None
        m = re.search(r"/(\d+)$", r.headers.get("content-range", ""))
        if m:
            total = int(m.group(1))
        elif r.status == 200 and r.headers.get("content-length", "").isdigit():
            total = int(r.headers["content-length"])
        payload = classify_magic(r.body)
        ctype = r.headers.get("content-type", "")
        ok = r.status in (200, 206) and payload not in ("xml/html-text", "unknown")
        out = {
            "url": url,
            "ok": ok,
            "status": r.status,
            "content_type": ctype,
            "total_size_bytes": total,
            "payload_type": payload,
            "first_bytes_hex": r.body[:8].hex(),
            "access": "file",
            "note": (
                "looks like a text/error page, not a data file"
                if payload == "xml/html-text"
                else ""
            ),
        }
        if payload in ("json-text", "xml/html-text") and r.status in (200, 206):
            out.update(self._probe_service(url, ctype))
        return out

    def _probe_service(self, url: str, content_type: str) -> dict:
        """16 bytes cannot tell a feature service from an error page - read enough to parse."""
        self._guard()
        r = self.transport("GET", url, {"User-Agent": "beegent-geofetch/1.0"},
                           config.PROBE_TEXT_BYTES)
        if r.error:
            return {}
        shape = classify_api_shape(r.body, content_type)
        if not shape:
            return {"ok": False,
                    "note": "text payload, but not a recognisable feature collection"}
        api = shape.pop("paged")
        # Boundary polygons dwarf the read budget, so say when the count is only a floor.
        cut = len(r.body) >= config.PROBE_TEXT_BYTES and not shape["count_is_exact"]
        note = (f"only the first {config.PROBE_TEXT_BYTES:,} bytes were read - feature_count "
                "is a FLOOR, not the dataset size; do not reject this endpoint as too small"
                ) if cut else ""
        # A page's content-length is not the dataset's size, and must not read like it.
        return {**shape, "ok": True, "note": note, "access": "api" if api else "file",
                **({"total_size_bytes": None} if api else {})}
