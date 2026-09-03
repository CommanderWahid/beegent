#!/usr/bin/env python3
"""The deterministic web layer: magic-byte sniffing, HTML reduction, search, probing."""

import pytest

from beegent import config
from beegent.web_tools import WebTools, classify_magic, summarize_html

from tests.conftest import (API_ERROR_URL, DEFAULT_PAGES, ED_2025, ESRI_QUERY_URL, FEED,
                            FILE_SIZE, FILE_URL, OAPIF_EMPTY, OAPIF_HUGE, OAPIF_ITEMS, OAPIF_NO_COUNT,
                            PORTAL, PORTAL_HTML, WFS_CAPS_URL)

# --- magic bytes ---


def test_signatures():
    cases = {b"PAR1xxxx": "parquet", b"PK\x03\x04rest": "zip",
             b"SQLite format 3\x00": "sqlite/geopackage",
             b"%PDF-1.7": "pdf", b"\x1f\x8bxx": "gzip",
             b"  <feed>": "xml/html-text", b'{"a":1}': "json-text",
             b"\x00\x00\x00": "unknown"}
    for raw, expected in cases.items():
        assert classify_magic(raw) == expected, raw


# --- HTML reduction ---


def test_extracts_link_tags_and_flags_js_shell():
    out = summarize_html(PORTAL_HTML.encode(), PORTAL)
    assert out["looks_like_js_app_shell"]
    assert FEED in [link["href"] for link in out["links"]]


def test_resolves_relative_anchors():
    html = b'<html><body><a href="/dl/file.zip">download</a></body></html>'
    out = summarize_html(html, "https://x.example/page")
    assert out["links"][0]["href"] == "https://x.example/dl/file.zip"
    assert out["links"][0]["text"] == "download"


def test_script_src_surfaces_in_links():
    out = summarize_html(PORTAL_HTML.encode(), PORTAL)
    script_links = [link for link in out["links"] if link["text"] == "<script src>"]
    assert script_links[0]["href"] == "https://geo.atlantis.example/static/app.bundle.js"


# --- the tools themselves ---


def test_probe_parses_content_range_and_magic(tools):
    probe = tools.probe_url(FILE_URL)
    assert probe["ok"]
    assert probe["status"] == 206
    assert probe["payload_type"] == "parquet"
    assert probe["total_size_bytes"] == FILE_SIZE


def test_probe_flags_html_error_page(tools):
    probe = tools.probe_url("https://api.atlantis.example/nope")
    assert not probe["ok"]
    assert "text" in probe["payload_type"]


def test_fetch_page_returns_raw_xml(tools):
    page = tools.fetch_page(FEED)
    assert "<feed" in page["body"]
    assert "links" not in page


def test_urls_found_surfaces_urls_from_xml_body(tools):
    assert FILE_URL in tools.fetch_page(ED_2025)["urls_found"]


def test_urls_found_excludes_urls_already_in_links(tools):
    """The two lists used to overlap, so a page paid twice for the same URL."""
    page = tools.fetch_page(PORTAL)
    hrefs = [link["href"] for link in page["links"]]
    assert FEED in hrefs, "the atom <link> still reaches the model via links"
    assert FEED not in page["urls_found"], "and is not duplicated in urls_found"
    assert PORTAL not in page["urls_found"], "nor is the page's own url"


def test_urls_found_still_surfaces_what_links_cannot(make_tools):
    """Its whole value: URLs in prose, XML or a JS bundle that no anchor would show."""
    buried = "https://api.atlantis.example/hidden/v2/catalog.json"
    html = (f'<html><body><a href="{FEED}">feed</a>'
            f'<p>see also {buried} for the raw catalogue</p></body></html>')
    pages = dict(DEFAULT_PAGES)
    pages["https://x.example/p"] = (200, "text/html", html)
    page = make_tools(pages=pages).fetch_page("https://x.example/p")
    assert buried in page["urls_found"], "prose-only URL must survive"
    assert FEED not in page["urls_found"], "but the anchor must not duplicate"


def test_web_search_parses_ddg_results_and_decodes_redirects(make_tools):
    from urllib.parse import quote_plus
    query = "atlantis landcover download parquet"
    ddg_url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    ddg_html = """<html><body>
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fapi.atlantis.example%2Fdl%2Fresource%2FLANDCOVER&rut=abc">Atlantis download service</a>
      <a href="https://docs.atlantis.example/landcover">LANDCOVER docs</a>
    </body></html>"""
    pages = dict(DEFAULT_PAGES)
    pages[ddg_url] = (200, "text/html", ddg_html)
    out = make_tools(pages=pages).web_search(query)
    urls = [r["url"] for r in out["results"]]
    assert FEED in urls  # uddg= redirect decoded
    assert "https://docs.atlantis.example/landcover" in urls
    assert all("duckduckgo.com" not in u for u in urls)


def test_web_search_falls_back_to_next_engine_on_challenge_page(make_tools):
    from urllib.parse import quote_plus
    query = "atlantis landcover download"
    q = quote_plus(query)
    challenge = "<html><body>anomaly detected, complete the challenge</body></html>"
    lite_html = ('<html><body><a href="https://api.atlantis.example/dl/'
                 'resource/LANDCOVER">Atlantis DL service</a></body></html>')
    pages = dict(DEFAULT_PAGES)
    pages[f"https://html.duckduckgo.com/html/?q={q}"] = (200, "text/html", challenge)
    pages[f"https://lite.duckduckgo.com/lite/?q={q}"] = (200, "text/html", lite_html)
    out = make_tools(pages=pages).web_search(query)
    assert out["engine"] == "ddg-lite"
    assert out["results"][0]["url"] == FEED


def test_web_search_total_failure_carries_bot_block_note(tools):
    out = tools.web_search("anything")  # no engine URLs routed
    assert out["results"] == []
    assert "bot-blocking" in out["note"]
    assert len(out["engines_tried"]) == 3


def test_bing_redirect_href_is_base64_decoded():
    import base64
    target = "https://api.atlantis.example/dl"
    b64 = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    href = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{b64}&ntb=1"
    assert WebTools._decode_result_href(href) == target
    assert WebTools._decode_result_href("https://x.example/a") == "https://x.example/a"


def test_request_budget_enforced(tools):
    tools.max_requests = 2
    tools.fetch_page(PORTAL)
    tools.fetch_page(FEED)
    with pytest.raises(RuntimeError):
        tools.fetch_page(ED_2025)


# --- feature services: 16 magic bytes cannot tell one from an error page ---


def test_oapif_reports_an_exact_count_from_number_matched(tools):
    """numberMatched sits after the features array, so this is the whole point of the 2nd read."""
    probe = tools.probe_url(OAPIF_ITEMS)
    assert probe["access"] == "api"
    assert probe["shape"] == "geojson_featurecollection"
    assert probe["feature_count"] == 4212
    assert probe["count_is_exact"]
    assert probe["geometry_type"] == "Polygon"


def test_a_page_count_is_reported_as_not_exact(tools):
    """Without numberMatched, the page length is a lower bound and must say so."""
    probe = tools.probe_url(OAPIF_NO_COUNT)
    assert probe["feature_count"] == 2
    assert not probe["count_is_exact"], "a page count must never look like a total"
    assert probe["geometry_type"] == "LineString"


def test_total_size_bytes_is_null_for_a_service(tools):
    """One page's content-length is not the dataset's size."""
    assert tools.probe_url(OAPIF_ITEMS)["total_size_bytes"] is None


def test_an_error_served_with_http_200_is_not_data(tools):
    """The case a leading '{' used to wave through."""
    probe = tools.probe_url(API_ERROR_URL)
    assert probe["status"] == 200
    assert not probe["ok"]
    assert probe.get("shape") is None
    assert probe["access"] == "file", "nothing identified it as a service"


def test_an_empty_feature_collection_is_still_parsed(tools):
    """Recognised, counted at zero - the report guardrail is what rejects it."""
    probe = tools.probe_url(OAPIF_EMPTY)
    assert probe["shape"] == "geojson_featurecollection"
    assert probe["feature_count"] == 0


def test_wfs_capabilities_is_recognised_as_metadata(tools):
    probe = tools.probe_url(WFS_CAPS_URL)
    assert probe["shape"] == "wfs_capabilities"
    assert probe["feature_count"] == 0


def test_esri_json_maps_its_geometry_names_to_ogc(tools):
    probe = tools.probe_url(ESRI_QUERY_URL)
    assert probe["shape"] == "esrijson_featureset"
    assert probe["geometry_type"] == "Polygon", "esriGeometryPolygon must be normalised"
    assert not probe["count_is_exact"], "exceededTransferLimit means the count is capped"


def test_a_static_geojson_file_is_not_an_api(make_tools):
    """A plain FeatureCollection with no service metadata stays access=file."""
    plain = ('{"type":"FeatureCollection","features":[{"type":"Feature",'
             '"geometry":{"type":"Point","coordinates":[0,0]},"properties":{}}]}')
    pages = dict(DEFAULT_PAGES)
    pages["https://x.example/data.geojson"] = (200, "application/geo+json", plain)
    probe = make_tools(pages=pages).probe_url("https://x.example/data.geojson")
    assert probe["access"] == "file"
    assert probe["feature_count"] == 1
    assert probe["total_size_bytes"] is not None, "a file keeps its size"


def test_a_truncated_document_counts_what_it_can_see(monkeypatch, tools):
    """Cut mid-document, the tail metadata is gone - a lower bound is the honest answer."""
    monkeypatch.setattr(config, "PROBE_TEXT_BYTES", 200)
    probe = tools.probe_url(OAPIF_ITEMS)
    assert probe["shape"] == "geojson_featurecollection"
    assert not probe["count_is_exact"]
    assert probe["feature_count"] < 4212


def test_a_truncated_service_is_still_recognised_as_an_api(monkeypatch, tools):
    """Measured on PDOK: 5 boundary polygons are 379 KB, so the tail metadata is never read.

    numberReturned sits after the features array and is lost, but OGC API-Features puts
    "links" before it - which is what must decide access.
    """
    monkeypatch.setattr(config, "PROBE_TEXT_BYTES", 6_000)
    probe = tools.probe_url(OAPIF_HUGE)
    assert probe["shape"] == "geojson_featurecollection"
    assert probe["access"] == "api", "a cut body must not demote a service to a file"
    assert probe["feature_count"] > 0


def test_a_floor_count_says_so_in_the_note(monkeypatch, tools):
    """A model reading "1 feature" would reject a national dataset as too small."""
    monkeypatch.setattr(config, "PROBE_TEXT_BYTES", 6_000)
    probe = tools.probe_url(OAPIF_HUGE)
    assert not probe["count_is_exact"]
    assert "FLOOR" in probe["note"]


def test_an_untruncated_service_carries_no_floor_note(tools):
    """The note must appear only when the read was actually cut short."""
    probe = tools.probe_url(OAPIF_ITEMS)
    assert probe["count_is_exact"]
    assert probe["note"] == ""
