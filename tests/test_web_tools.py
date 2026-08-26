#!/usr/bin/env python3
"""The deterministic web layer: magic-byte sniffing, HTML reduction, search, probing."""

import unittest

from beegent.web_tools import WebTools, classify_magic, summarize_html

from tests.fixtures import (DEFAULT_PAGES, ED_2025, FEED, FILE_SIZE, FILE_URL, PORTAL,
                            PORTAL_HTML, make_tools)


class TestMagic(unittest.TestCase):
    def test_signatures(self):
        cases = {b"PAR1xxxx": "parquet", b"PK\x03\x04rest": "zip",
                 b"SQLite format 3\x00": "sqlite/geopackage",
                 b"%PDF-1.7": "pdf", b"\x1f\x8bxx": "gzip",
                 b"  <feed>": "xml/html-text", b'{"a":1}': "json-text",
                 b"\x00\x00\x00": "unknown"}
        for raw, expected in cases.items():
            self.assertEqual(classify_magic(raw), expected, raw)


class TestHtmlSummary(unittest.TestCase):
    def test_extracts_link_tags_and_flags_js_shell(self):
        out = summarize_html(PORTAL_HTML.encode(), PORTAL)
        self.assertTrue(out["looks_like_js_app_shell"])
        self.assertIn(FEED, [link["href"] for link in out["links"]])

    def test_resolves_relative_anchors(self):
        html = b'<html><body><a href="/dl/file.zip">download</a></body></html>'
        out = summarize_html(html, "https://x.example/page")
        self.assertEqual(out["links"][0]["href"], "https://x.example/dl/file.zip")
        self.assertEqual(out["links"][0]["text"], "download")

    def test_script_src_surfaces_in_links(self):
        out = summarize_html(PORTAL_HTML.encode(), PORTAL)
        script_links = [link for link in out["links"] if link["text"] == "<script src>"]
        self.assertEqual(script_links[0]["href"],
                         "https://geo.atlantis.example/static/app.bundle.js")


class TestWebTools(unittest.TestCase):
    def test_probe_parses_content_range_and_magic(self):
        probe = make_tools().probe_url(FILE_URL)
        self.assertTrue(probe["ok"])
        self.assertEqual(probe["status"], 206)
        self.assertEqual(probe["payload_type"], "parquet")
        self.assertEqual(probe["total_size_bytes"], FILE_SIZE)

    def test_probe_flags_html_error_page(self):
        probe = make_tools().probe_url("https://api.atlantis.example/nope")
        self.assertFalse(probe["ok"])
        self.assertIn("text", probe["payload_type"])

    def test_fetch_page_returns_raw_xml(self):
        page = make_tools().fetch_page(FEED)
        self.assertIn("<feed", page["body"])
        self.assertNotIn("links", page)

    def test_urls_found_surfaces_urls_from_xml_body(self):
        self.assertIn(FILE_URL, make_tools().fetch_page(ED_2025)["urls_found"])

    def test_urls_found_surfaces_urls_from_html(self):
        self.assertIn(FEED, make_tools().fetch_page(PORTAL)["urls_found"])

    def test_web_search_parses_ddg_results_and_decodes_redirects(self):
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
        self.assertIn(FEED, urls)  # uddg= redirect decoded
        self.assertIn("https://docs.atlantis.example/landcover", urls)
        self.assertTrue(all("duckduckgo.com" not in u for u in urls))

    def test_web_search_falls_back_to_next_engine_on_challenge_page(self):
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
        self.assertEqual(out["engine"], "ddg-lite")
        self.assertEqual(out["results"][0]["url"], FEED)

    def test_web_search_total_failure_carries_bot_block_note(self):
        out = make_tools().web_search("anything")  # no engine URLs routed
        self.assertEqual(out["results"], [])
        self.assertIn("bot-blocking", out["note"])
        self.assertEqual(len(out["engines_tried"]), 3)

    def test_bing_redirect_href_is_base64_decoded(self):
        import base64
        target = "https://api.atlantis.example/dl"
        b64 = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{b64}&ntb=1"
        self.assertEqual(WebTools._decode_result_href(href), target)
        self.assertEqual(WebTools._decode_result_href("https://x.example/a"),
                         "https://x.example/a")

    def test_request_budget_enforced(self):
        tools = make_tools()
        tools.max_requests = 2
        tools.fetch_page(PORTAL)
        tools.fetch_page(FEED)
        with self.assertRaises(RuntimeError):
            tools.fetch_page(ED_2025)


if __name__ == "__main__":
    unittest.main(verbosity=2)
