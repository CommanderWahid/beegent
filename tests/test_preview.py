#!/usr/bin/env python3
"""The map preview byte pipe: what it refuses, what it caches, and when it refetches."""

import os
import time

import pytest

from beegent import preview
from tests.conftest import PREVIEW_GEOJSON, PREVIEW_GPKG, fake_stream_transport

GPKG_URL = "https://data.example/boundaries.gpkg"
API_URL = "https://data.example/collections/gebieden/items"


@pytest.fixture
def previews(tmp_path, monkeypatch):
    """A cache dir of this test's own, and the defaults pinned so a .env cannot move them."""
    monkeypatch.setattr("beegent.config.PREVIEW_DIR", str(tmp_path))
    monkeypatch.setattr("beegent.config.PREVIEW_CACHE_DAYS", 7)
    monkeypatch.setattr("beegent.config.PREVIEW_MAX_BYTES", 50_000_000)
    return tmp_path


def _row(payload_type="sqlite/geopackage", shape="", access="file", size=None,
         url=GPKG_URL, status="ok", last_verified="", uid="u1", content_type=""):
    """A links row as SqliteStore.link() returns it: claim and verification already decoded."""
    return {
        "link_uid": uid, "country": "Atlantis", "dataset": "administrative boundaries",
        "url": "https://portal.example/dataset", "resource_url": url,
        "confidence": 0.95, "last_verified": last_verified, "status": status, "claim": {},
        "verification": {"ok": True, "status": 206, "payload_type": payload_type,
                         "shape": shape, "access": access, "total_size_bytes": size,
                         "content_type": content_type},
    }


def _geojson_row(**kw):
    kw.setdefault("payload_type", "json-text")
    kw.setdefault("shape", "geojson_featurecollection")
    return _row(**kw)


# --- asking the service for WGS84 -------------------------------------------


@pytest.mark.parametrize("shape,expected", [
    ("geojson_featurecollection", "crs=http%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FOGC%2F1.3%2FCRS84"),
    ("esrijson_featureset", "outSR=4326"),
    ("wfs_featurecollection", "srsName=urn%3Aogc%3Adef%3Acrs%3AOGC%3A1.3%3ACRS84"),
])
def test_an_api_row_is_asked_for_lon_lat(shape, expected):
    """deck.gl cannot reproject, so the one free lever is the request itself."""
    assert expected in preview.wgs84_params(API_URL, shape, "api")


def test_a_file_row_is_never_rewritten():
    """A static file has no such lever; the client sanity-checks its coordinates instead."""
    assert preview.wgs84_params(GPKG_URL, "geojson_featurecollection", "file") == GPKG_URL


def test_an_unrecognised_shape_is_left_alone():
    assert preview.wgs84_params(API_URL, "", "api") == API_URL


def test_a_parameter_already_present_is_not_restated():
    """Stacking a second outSR= would be a guess overriding what was actually verified."""
    url = f"{API_URL}?outSR=28992&f=json"
    assert preview.wgs84_params(url, "esrijson_featureset", "api") == url


# --- where the bytes land ----------------------------------------------------


@pytest.mark.parametrize("payload,ctype,shape,ext", [
    ("sqlite/geopackage", "", "", ".gpkg"),
    ("parquet", "", "", ".parquet"),
    ("json-text", "", "geojson_featurecollection", ".geojson"),
    ("json-text", "", "esrijson_featureset", ".json"),
    ("", "application/geo+json", "", ".geojson"),
    ("", "application/json; charset=utf-8", "", ".json"),
    ("", "", "", ".bin"),
])
def test_the_cached_name_says_what_it_holds(previews, payload, ctype, shape, ext):
    """Cosmetic, for a human reading `ls` - nothing ever dispatches on this extension."""
    assert preview.cache_path("u1", payload, ctype, shape).endswith(f"u1{ext}")


# --- refusals that cost nothing ----------------------------------------------


def test_an_unsupported_payload_is_refused_without_a_request(previews):
    calls = []
    out = preview.ensure_cached(_row(payload_type="pdf"),
                                fake_stream_transport({}, calls))
    assert out.reason == "unsupported" and "pdf" in out.detail
    assert calls == []


def test_a_json_payload_that_is_not_a_feature_collection_is_refused(previews):
    """json-text alone is not enough: the probe must have recognised a feature shape."""
    out = preview.ensure_cached(_row(payload_type="json-text", shape=""),
                                fake_stream_transport({}))
    assert out.reason == "unsupported"


def test_an_oversized_payload_is_refused_from_the_stored_size_alone(previews, monkeypatch):
    """The probe already measured it, so the refusal opens no socket at all."""
    monkeypatch.setattr("beegent.config.PREVIEW_MAX_BYTES", 1_000)
    calls = []
    out = preview.ensure_cached(_row(size=14_495_744), fake_stream_transport({}, calls))
    assert out.reason == "too_large"
    assert "14.5 MB" in out.detail and "0.0 MB" in out.detail
    assert calls == []  # the whole point: refused BEFORE the request, not after


def test_previews_can_be_switched_off_entirely(previews, monkeypatch):
    monkeypatch.setattr("beegent.config.PREVIEW_DIR", "")
    out = preview.ensure_cached(_row(), fake_stream_transport({}))
    assert out.reason == "disabled" and "PREVIEW_DIR" in out.detail


# --- what may be fetched -----------------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://data.example/x.gpkg",
                                 "data:application/json,{}"])
def test_a_scheme_outside_http_is_refused_before_any_socket(previews, url):
    calls = []
    out = preview.ensure_cached(_row(url=url), fake_stream_transport({}, calls))
    assert out.reason == "blocked"
    assert calls == []


def test_a_redirect_is_followed(previews):
    hop = "https://cdn.example/real.gpkg"
    pages = {GPKG_URL: (302, {"location": hop}, b""),
             hop: (200, {}, PREVIEW_GPKG)}
    out = preview.ensure_cached(_row(), fake_stream_transport(pages))
    assert out.reason == "" and out.bytes_written == len(PREVIEW_GPKG)


def test_a_redirect_to_a_forbidden_scheme_is_caught_mid_chain(previews):
    """requests would follow this transparently, so the allowlist must run on EVERY hop."""
    pages = {GPKG_URL: (302, {"location": "file:///etc/passwd"}, b"")}
    out = preview.ensure_cached(_row(), fake_stream_transport(pages))
    assert out.reason == "blocked" and "file:" in out.detail


def test_a_redirect_loop_is_abandoned(previews):
    calls = []
    pages = {GPKG_URL: (302, {"location": GPKG_URL}, b"")}
    out = preview.ensure_cached(_row(), fake_stream_transport(pages, calls))
    assert out.reason == "blocked" and "redirects" in out.detail
    assert len(calls) == preview._MAX_REDIRECTS + 1


def test_an_upstream_error_is_reported_with_its_status(previews):
    out = preview.ensure_cached(_row(), fake_stream_transport({GPKG_URL: (403, {}, b"")}))
    assert out.reason == "http" and "403" in out.detail


def test_a_wgs84_parameter_that_breaks_the_url_is_dropped_and_retried(previews):
    """A guessed parameter that breaks a working URL is worse than an unprojected map."""
    asked = preview.wgs84_params(API_URL, "geojson_featurecollection", "api")
    calls = []
    pages = {asked: (400, {}, b""), API_URL: (200, {}, PREVIEW_GEOJSON)}
    out = preview.ensure_cached(_geojson_row(url=API_URL, access="api"),
                                fake_stream_transport(pages, calls))
    assert out.reason == "" and out.fetched_url == API_URL
    assert calls == [asked, API_URL]


# --- the cap, and never a truncated cache entry ------------------------------


def test_a_declared_length_over_the_cap_is_refused_before_the_body(previews, monkeypatch):
    """Covers an `api` row, whose total_size_bytes is None by design."""
    monkeypatch.setattr("beegent.config.PREVIEW_MAX_BYTES", 10)
    pages = {GPKG_URL: (200, {"content-length": "999999"}, PREVIEW_GPKG)}
    out = preview.ensure_cached(_row(size=None), fake_stream_transport(pages))
    assert out.reason == "too_large"
    assert not os.listdir(previews)


def test_a_stream_over_the_cap_leaves_nothing_behind(previews, monkeypatch):
    """The backstop for a chunked response that declared no length at all.

    A half-written .gpkg kept at the real filename would be served as a valid cache entry
    forever - so the partial must not survive, under either name.
    """
    monkeypatch.setattr("beegent.config.PREVIEW_MAX_BYTES", 10)
    pages = {GPKG_URL: (200, {}, PREVIEW_GPKG)}
    out = preview.ensure_cached(_row(size=None), fake_stream_transport(pages))
    assert out.reason == "too_large"
    assert os.listdir(previews) == []  # no .gpkg, and no .part either


def test_a_successful_fetch_writes_exactly_what_was_served(previews):
    """Exactly as received: no conversion, no normalisation, byte for byte."""
    pages = {GPKG_URL: (200, {"content-type": "application/geopackage+sqlite3"}, PREVIEW_GPKG)}
    out = preview.ensure_cached(_row(), fake_stream_transport(pages))
    assert out.reason == ""
    assert open(out.path, "rb").read() == PREVIEW_GPKG
    assert out.content_type == "application/geopackage+sqlite3"
    assert os.listdir(previews) == ["u1.gpkg"]


# --- when a cached payload still answers -------------------------------------


def _cache(previews, name="u1.gpkg", body=PREVIEW_GPKG):
    path = previews / name
    path.write_bytes(body)
    return path


def test_a_cached_payload_is_served_without_a_request(previews):
    _cache(previews)
    calls = []
    out = preview.ensure_cached(_row(), fake_stream_transport({}, calls))
    assert out.cached and out.bytes_written == len(PREVIEW_GPKG)
    assert calls == []


def test_a_cache_older_than_the_window_is_refetched(previews):
    path = _cache(previews)
    old = time.time() - 30 * 86_400
    os.utime(path, (old, old))
    out = preview.ensure_cached(_row(), fake_stream_transport({GPKG_URL: (200, {}, b"fresh")}))
    assert not out.cached and open(out.path, "rb").read() == b"fresh"


def test_a_cache_written_before_the_link_was_reverified_is_refetched(previews):
    """The rule that keeps the cache FOLLOWING the pipeline: a re-probe may be a new edition."""
    path = _cache(previews)
    old = time.time() - 3_600
    os.utime(path, (old, old))
    just_now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    out = preview.ensure_cached(_row(last_verified=just_now),
                                fake_stream_transport({GPKG_URL: (200, {}, b"newer")}))
    assert not out.cached and open(out.path, "rb").read() == b"newer"


def test_zero_days_always_refetches(previews, monkeypatch):
    """The documented off switch, and it has to work from the environment too."""
    monkeypatch.setattr("beegent.config.PREVIEW_CACHE_DAYS", 0)
    _cache(previews)
    calls = []
    out = preview.ensure_cached(_row(), fake_stream_transport({GPKG_URL: (200, {}, b"again")}, calls))
    assert not out.cached and calls == [GPKG_URL]


def test_an_empty_cache_file_is_removed_and_refetched(previews):
    _cache(previews, body=b"")
    out = preview.ensure_cached(_row(), fake_stream_transport({GPKG_URL: (200, {}, PREVIEW_GPKG)}))
    assert not out.cached and out.bytes_written == len(PREVIEW_GPKG)


def test_a_rotten_link_is_not_dialled_again(previews):
    calls = []
    out = preview.ensure_cached(_row(status="ko"), fake_stream_transport({}, calls))
    assert out.reason == "rot"
    assert calls == []


def test_a_rotten_link_still_serves_what_was_cached_while_it_lived(previews):
    """A cached copy is a real artifact of the last good fetch; the card already says it died."""
    _cache(previews)
    out = preview.ensure_cached(_row(status="ko"), fake_stream_transport({}))
    assert out.cached and out.reason == ""


def test_dropping_a_cached_payload_reports_whether_there_was_one(previews):
    _cache(previews)
    assert preview.drop_cached("u1") is True
    assert preview.drop_cached("u1") is False
    assert os.listdir(previews) == []
