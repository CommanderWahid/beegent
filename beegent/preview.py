#!/usr/bin/env python3
"""Fetch the bytes behind a verified link, under a cap, and cache them as received.

A guarded byte pipe and NOTHING more: it never parses a geospatial format, never converts
one, and never writes back into `verification`. The browser does the parsing; this module
only decides whether a fetch is allowed, bounds what it costs, and keeps the result.

The preview is a visual aid, never evidence. That is what lets it add a query parameter to
ask a service for WGS84 - a rewrite `probe_url()` may never do, because the probe result IS
the verification guarantee. See docs/output-schema.md.
"""

import calendar
import glob
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from beegent import config

_log = logging.getLogger(__name__)

# NOT overridable: a safety floor, not a preference. Five hops reaches any real download.
_MAX_REDIRECTS = 5
_REDIRECTS = (301, 302, 303, 307, 308)
_ALLOWED_SCHEMES = ("http", "https")
_CHUNK = 65_536

# What the browser can draw. The server does not parse, but it does refuse early: a
# hand-crafted request must not spend 50MB of budget on a PDF the UI already greys out.
_MAPPABLE_SHAPES = ("geojson_featurecollection", "esrijson_featureset")

_EXT_BY_PAYLOAD = {
    "parquet": ".parquet", "sqlite/geopackage": ".gpkg", "zip": ".zip", "gzip": ".gz",
    "tiff/geotiff": ".tif", "pdf": ".pdf", "png": ".png",
    "json-text": ".json", "xml/html-text": ".xml",
}
_EXT_BY_CONTENT = {
    "application/geo+json": ".geojson", "application/json": ".json",
    "application/geopackage+sqlite3": ".gpkg", "application/x-sqlite3": ".gpkg",
    "application/xml": ".xml", "text/xml": ".xml", "application/zip": ".zip",
}
# What to send back when the row recorded no content type of its own.
_MIME_BY_PAYLOAD = {
    "parquet": "application/vnd.apache.parquet",
    "sqlite/geopackage": "application/geopackage+sqlite3",
    "zip": "application/zip", "gzip": "application/gzip",
    "json-text": "application/json", "xml/html-text": "application/xml",
}


@dataclass
class Preview:
    """What a preview attempt produced. An empty `reason` means the bytes are on disk."""

    path: str = ""
    bytes_written: int = 0
    cached: bool = False  # served from disk: no request was made
    content_type: str = ""
    fetched_url: str = ""  # what was actually requested, after any WGS84 rewrite
    reason: str = ""  # "" | disabled | unsupported | rot | too_large | blocked | http | network
    detail: str = ""


class Response:
    """One HTTP hop. `raw` is a streaming requests response; `body` is what a test hands back."""

    def __init__(self, status=0, headers=None, url="", error="", body=b"", raw=None):
        self.status = status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.url = url
        self.error = error
        self._body = body
        self._raw = raw

    def stream(self, chunk_size=_CHUNK):
        if self._raw is not None:
            yield from self._raw.iter_content(chunk_size=chunk_size)
            return
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        if self._raw is not None:
            self._raw.close()


def streaming_transport(url: str, headers: dict, timeout) -> Response:
    """The injectable seam, exactly as WebTools takes one - so the suite stays offline.

    Redirects are NOT followed here: ensure_cached() walks them itself, so the scheme
    allowlist is enforced on every hop rather than only on the first.
    """
    import requests

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=timeout,
                            allow_redirects=False)
    except requests.RequestException as exc:
        return Response(error=f"{type(exc).__name__}: {exc}", url=url)
    return Response(resp.status_code, dict(resp.headers), resp.url, raw=resp)


def wgs84_params(url: str, shape: str, access: str) -> str:
    """Ask the SERVICE for lon/lat, since deck.gl cannot reproject. Never touches a payload.

    CRS84 rather than EPSG:4326 deliberately: 4326 is officially lat/lon while GeoJSON is
    lon/lat, so a strictly conformant server hands back Dutch data that draws off Somalia.
    A parameter already present is never restated - the stored URL wins.
    """
    if access != "api":
        return url  # a static file has no such lever; the client sanity-checks it instead
    add = {}
    if shape == "geojson_featurecollection":
        add["crs"] = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
    elif shape == "esrijson_featureset":
        add["outSR"] = "4326"
        add["f"] = "geojson"
    elif shape == "wfs_featurecollection":
        add["srsName"] = "urn:ogc:def:crs:OGC:1.3:CRS84"
    else:
        return url
    parts = urlparse(url)
    have = parse_qs(parts.query, keep_blank_values=True)
    fresh = {k: v for k, v in add.items() if k not in have}
    if not fresh:
        return url
    query = parts.query + ("&" if parts.query else "") + urlencode(fresh)
    return urlunparse(parts._replace(query=query))


def mappable(verification: dict) -> bool:
    """Whether a browser could draw this at all. A format check, never a parse."""
    payload = (verification or {}).get("payload_type", "")
    if payload == "sqlite/geopackage":
        return True
    return payload == "json-text" and verification.get("shape") in _MAPPABLE_SHAPES


def cache_path(link_uid: str, payload_type: str, content_type: str, shape: str = "") -> str:
    """Where the bytes land. The extension is COSMETIC - for a human reading `ls`.

    Nothing dispatches on it: the browser keys off verification.payload_type, which the
    harness measured from magic bytes. Do not turn this into a format decision.
    """
    ext = ""
    if payload_type == "json-text" and shape == "geojson_featurecollection":
        ext = ".geojson"
    if not ext:
        ext = _EXT_BY_PAYLOAD.get(payload_type, "")
    if not ext:
        ext = _EXT_BY_CONTENT.get((content_type or "").split(";")[0].strip().lower(), "")
    return os.path.join(config.PREVIEW_DIR, f"{link_uid}{ext or '.bin'}")


def _allowed(url: str) -> str:
    """Empty if this URL may be fetched, else why not.

    Kept out of normalize_url(), which is a comparison form used when WRITING
    resource_url_norm - validating there would change stored dedup keys.
    """
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"{scheme or 'relative'}: is not a fetchable scheme"
    return ""


def _cached_file(link_uid: str) -> str:
    """The cached payload for this link, whatever extension it was written under."""
    hits = [p for p in glob.glob(os.path.join(config.PREVIEW_DIR, f"{link_uid}.*"))
            if not p.endswith(".part")]
    return hits[0] if hits else ""


def _fresh(path: str, last_verified: str) -> bool:
    """Age, provenance and emptiness - the three ways a cached file stops being the answer."""
    try:
        stat = os.stat(path)
    except OSError:
        return False
    if not stat.st_size:
        os.unlink(path)  # a zero-byte entry is a failed write, not an answer
        return False
    if not config.PREVIEW_CACHE_DAYS:
        return False  # 0 is the documented off switch: always refetch
    if time.time() - stat.st_mtime > config.PREVIEW_CACHE_DAYS * 86_400:
        return False
    # The link was re-probed since this was written, so it may now point at a new edition.
    # This is what keeps the cache FOLLOWING the pipeline instead of drifting away from it.
    try:
        # store._now() writes UTC, so timegm - mktime would read it as local and drift.
        verified = calendar.timegm(time.strptime(last_verified[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return True  # no usable timestamp: age alone decides
    return stat.st_mtime >= verified


def _over_cap(size) -> bool:
    return isinstance(size, int) and size > config.PREVIEW_MAX_BYTES


def _mb(n) -> str:
    """Name the number the way a person would say it: a cap refusal has to be readable."""
    if not isinstance(n, int):
        return "an unstated size"
    return f"{n / 1_000_000_000:.1f} GB" if n >= 1_000_000_000 else f"{n / 1_000_000:.1f} MB"


class _TooLarge(Exception):
    """The mid-stream backstop, for a response that declared no length."""


def _fetch(url: str, transport) -> tuple:
    """Walk redirects ourselves, checking the scheme on EVERY hop.

    Returns (Response | None, reason, detail). Letting requests follow them would show us
    only hop 0, so the allowlist would be enforced once and then bypassed.
    """
    headers = {"User-Agent": "beegent-preview/1.0", "Accept": "*/*"}
    timeout = (10, config.PREVIEW_TIMEOUT)  # a scalar bounds neither connect nor total
    seen = url
    for _ in range(_MAX_REDIRECTS + 1):
        bad = _allowed(seen)
        if bad:
            return None, "blocked", bad
        resp = transport(seen, headers, timeout)
        if resp.error:
            return None, "network", resp.error
        if resp.status not in _REDIRECTS:
            return resp, "", ""
        target = resp.headers.get("location", "")
        resp.close()
        if not target:
            return None, "http", f"HTTP {resp.status} with no Location"
        seen = urljoin(seen, target)  # Location is allowed to be relative
    return None, "blocked", f"more than {_MAX_REDIRECTS} redirects"


def _write(resp: Response, dest: str) -> tuple:
    """Stream to a .part file and os.replace() it into place.

    Atomic on purpose: a truncated .gpkg left at the final name would be served as a valid
    cache entry forever. Nothing partial ever gets the real filename.
    """
    part = f"{dest}.{os.getpid()}.part"
    written = 0
    try:
        with open(part, "wb") as fh:
            for chunk in resp.stream():
                if written + len(chunk) > config.PREVIEW_MAX_BYTES:
                    raise _TooLarge()
                fh.write(chunk)
                written += len(chunk)
        os.replace(part, dest)
        return written, "", ""
    except _TooLarge:
        return 0, "too_large", (f"payload exceeds PREVIEW_MAX_BYTES "
                                f"({_mb(config.PREVIEW_MAX_BYTES)})")
    except OSError as exc:
        return 0, "network", f"{type(exc).__name__}: {exc}"
    finally:
        resp.close()
        if os.path.exists(part):
            os.unlink(part)


def ensure_cached(row: dict, transport=streaming_transport) -> Preview:
    """Fetch (or reuse) the bytes behind one stored link. Takes the ROW, never a URL.

    That is the SSRF boundary: the set of URLs this can be made to fetch is exactly the set
    the pipeline already probed and stored. A caller cannot point it anywhere new.
    """
    if not config.PREVIEW_DIR:
        return Preview(reason="disabled", detail="previews are disabled: set PREVIEW_DIR")

    ver = row.get("verification") or {}
    if not mappable(ver):
        payload = ver.get("payload_type") or "an unrecognised payload"
        return Preview(reason="unsupported", detail=f"{payload} cannot be drawn on a map")

    link_uid = row["link_uid"]
    content_type = ver.get("content_type") or _MIME_BY_PAYLOAD.get(ver.get("payload_type"), "")
    dest = cache_path(link_uid, ver.get("payload_type", ""), content_type, ver.get("shape", ""))

    hit = _cached_file(link_uid)
    if hit and _fresh(hit, row.get("last_verified", "")):
        return Preview(path=hit, bytes_written=os.path.getsize(hit), cached=True,
                       content_type=content_type, fetched_url=row.get("resource_url", ""))

    # Rot is the store verdict, not ours: a cached copy is still a real artifact of the last
    # good fetch, but a dead link must not be dialled again to make a new one.
    if row.get("status") != "ok":
        return Preview(reason="rot",
                       detail="this link is marked no longer reachable and nothing is cached")

    declared = ver.get("total_size_bytes")
    if _over_cap(declared):  # free refusal: the probe measured it, so no socket is opened
        return Preview(reason="too_large",
                       detail=(f"payload is {_mb(declared)}; PREVIEW_MAX_BYTES is "
                               f"{_mb(config.PREVIEW_MAX_BYTES)}"))

    original = row["resource_url"]
    url = wgs84_params(original, ver.get("shape", ""), ver.get("access", ""))
    resp, reason, detail = _fetch(url, transport)
    # A guessed parameter that breaks a working URL is worse than an unprojected map.
    if resp is not None and 400 <= resp.status < 500 and url != original:
        _log.info(f"[preview] {resp.status} with a WGS84 parameter; retrying as stored")
        resp.close()
        url = original
        resp, reason, detail = _fetch(url, transport)
    if resp is None:
        return Preview(reason=reason, detail=detail, fetched_url=url)
    if resp.status >= 400 or resp.status == 0:
        resp.close()
        return Preview(reason="http", detail=f"upstream returned HTTP {resp.status}",
                       fetched_url=url)

    # The second free refusal: what the server itself says, before a byte is read. This is
    # the gate that covers an `api` row, whose total_size_bytes is None by design.
    try:
        length = int(resp.headers.get("content-length", ""))
    except ValueError:
        length = None
    if _over_cap(length):
        resp.close()
        return Preview(reason="too_large", fetched_url=url,
                       detail=(f"payload is {_mb(length)}; PREVIEW_MAX_BYTES is "
                               f"{_mb(config.PREVIEW_MAX_BYTES)}"))

    os.makedirs(config.PREVIEW_DIR, exist_ok=True)
    written, reason, detail = _write(resp, dest)
    if reason:
        return Preview(reason=reason, detail=detail, fetched_url=url)
    served = (resp.headers.get("content-type") or content_type).split(";")[0].strip()
    _log.info(f"[preview] {written} bytes cached for {link_uid}")
    return Preview(path=dest, bytes_written=written, content_type=served, fetched_url=url)


def drop_cached(link_uid: str) -> bool:
    """Forget one cached payload. Returns whether anything was there."""
    hit = _cached_file(link_uid) if config.PREVIEW_DIR else ""
    if not hit:
        return False
    os.unlink(hit)
    return True
