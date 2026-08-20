#!/usr/bin/env python3
"""Offline tests for the geofetch stage - no network, no API key, no LLM.

Three layers are tested:
  1. The deterministic tools (magic sniffing, HTML summarization, probing, search).
  2. The full agent loop, using a scripted fake `chat_tools` navigating a SYNTHETIC portal
     for a fictional country ("Atlantis") - deliberately not any real portal, which is what
     proves nothing real-world is hardcoded in the harness.
  3. The pipeline seam: resolve_angle() -> Candidate, and run.py's _rank().

    uv run python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest import mock

from core import config
from core.llm import strip_think
from core.pipeline.geofetch import (
    GeofetchAgent,
    _compact_history,
    coerce_report_args,
    extract_inline_tool_call,
    resolve_angle,
)
from core.pipeline.critic import needs_escalation
from core.run import _rank
from core.schemas import Candidate, SearchAngle, TokenUsage
from core.search_backends.web_tools import (
    HttpResult,
    WebTools,
    classify_magic,
    summarize_html,
)

# --------------------------------------------------------------------------- #
# Synthetic portal fixtures (fictional country, fictional dataset)
# --------------------------------------------------------------------------- #

PORTAL = "https://geo.atlantis.example/datasets/LANDCOVER"
FEED = "https://api.atlantis.example/dl/resource/LANDCOVER"
ED_2024 = f"{FEED}/LANDCOVER_PARQUET_ATL_2024-01-01"
ED_2025 = f"{FEED}/LANDCOVER_PARQUET_ATL_2025-07-01"
FILE_URL = ("https://api.atlantis.example/dl/download/LANDCOVER/"
            "LANDCOVER_PARQUET_ATL_2025-07-01/forest.parquet")
FILE_SIZE = 123_456_789

PORTAL_HTML = f"""<!doctype html><html><head>
<title>Atlantis Geoportal</title>
<link rel="alternate" type="application/atom+xml" href="{FEED}"/>
<script src="/static/app.bundle.js"></script>
</head><body><div id="root"></div>{'<!-- pad -->' * 300}</body></html>"""

FEED_XML = f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" pagecount="1">
  <entry><title>LANDCOVER_PARQUET_ATL_2024-01-01</title>
    <link rel="alternate" href="{ED_2024}"/><editionDate>2024-01-01</editionDate></entry>
  <entry><title>LANDCOVER_PARQUET_ATL_2025-07-01</title>
    <link rel="alternate" href="{ED_2025}"/><editionDate>2025-07-01</editionDate></entry>
</feed>"""

EDITION_XML = f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" pagecount="1">
  <entry><link rel="alternate" href="{FILE_URL}" length="{FILE_SIZE}"/>
    <content>d41d8cd98f00b204e9800998ecf8427e</content></entry>
</feed>"""

PARQUET_HEAD = b"PAR1" + b"\x00" * 12


def fake_transport(pages: dict, binaries: dict):
    """Build a transport closure over {url: (status, ctype, body)} pages and
    {url: total_size} binary files (which honour Range requests)."""
    def transport(method, url, headers, max_bytes):
        if url in binaries:
            total = binaries[url]
            if "range" in {k.lower() for k in headers}:
                return HttpResult(206, {
                    "content-type": "application/vnd.apache.parquet",
                    "content-range": f"bytes 0-15/{total}"},
                    PARQUET_HEAD, url)
            return HttpResult(200, {
                "content-type": "application/vnd.apache.parquet",
                "content-length": str(total)}, PARQUET_HEAD, url)
        if url in pages:
            status, ctype, body = pages[url]
            return HttpResult(status, {"content-type": ctype},
                              body.encode()[:max_bytes], url)
        return HttpResult(404, {"content-type": "text/html"},
                          b"<html>not found</html>", url)
    return transport


DEFAULT_PAGES = {
    PORTAL: (200, "text/html", PORTAL_HTML),
    FEED: (200, "application/atom+xml", FEED_XML),
    ED_2025: (200, "application/atom+xml", EDITION_XML),
}


def make_tools(pages=None, binaries=None):
    return WebTools(transport=fake_transport(pages or DEFAULT_PAGES,
                                             binaries or {FILE_URL: FILE_SIZE}))


# --------------------------------------------------------------------------- #
# Scripted LLM (shaped like the OpenAI SDK message chat_tools returns)
# --------------------------------------------------------------------------- #

_CALL_N = [0]


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, name, args):
        _CALL_N[0] += 1
        self.id = f"call_{_CALL_N[0]}"
        self.type = "function"
        self.function = _Fn(name, json.dumps(args))


class _Msg:
    """An assistant message with the attributes chat_tools' caller reads."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content or ""
        self.tool_calls = list(tool_calls or [])


class FakeLLM:
    """Scripted stand-in for core.llm.chat_tools: pops one (message, usage) pair per call
    and records the message history it was handed."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_messages = []
        self.chat_calls = 0

    def __call__(self, model, messages, tools):
        self.seen_messages = messages
        self.chat_calls += 1
        if not self.turns:
            raise AssertionError("FakeLLM script exhausted")
        return self.turns.pop(0), TokenUsage(prompt_tokens=100, completion_tokens=20)


def tool_turn(name, args, thought=None):
    return _Msg(content=thought, tool_calls=[_Call(name, args)])


GOOD_REPORT = {"found": True, "download_url": FILE_URL,
               "edition": "LANDCOVER_PARQUET_ATL_2025-07-01",
               "vintage_date": "2025-07-01", "file_size_bytes": FILE_SIZE,
               "confidence": "high",
               "evidence": [PORTAL, FEED, ED_2025, FILE_URL]}

HAPPY_PATH = [
    tool_turn("fetch_page", {"url": PORTAL}, "Reconnaissance."),
    tool_turn("fetch_page", {"url": FEED},
              "HTML is a JS shell; following the atom <link>."),
    tool_turn("fetch_page", {"url": ED_2025},
              "2025-07-01 is the latest edition."),
    tool_turn("probe_url", {"url": FILE_URL}, "Verifying before reporting."),
    tool_turn("report_result", GOOD_REPORT),
]


# --------------------------------------------------------------------------- #
# Tool-layer tests
# --------------------------------------------------------------------------- #

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


class TestStripThink(unittest.TestCase):
    def test_strip_think_removes_reasoning_blocks(self):
        raw = "<think>long hidden\nreasoning</think>Following the atom link."
        self.assertEqual(strip_think(raw), "Following the atom link.")
        self.assertEqual(strip_think(None), "")
        self.assertEqual(strip_think("no blocks here"), "no blocks here")

    def test_think_blocks_stripped_from_kept_history(self):
        turns = [tool_turn("fetch_page", {"url": PORTAL},
                           thought="<think>secret</think>recon")] + list(HAPPY_PATH)[1:]
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        assistants = [m for m in llm.seen_messages if m.get("role") == "assistant"]
        self.assertTrue(any(m["content"] == "recon" for m in assistants))
        self.assertFalse(any("<think>" in m.get("content", "") for m in assistants))


# --------------------------------------------------------------------------- #
# Agent-loop tests
# --------------------------------------------------------------------------- #

def run_agent(turns, tools=None, max_steps=10, fmt="GeoParquet"):
    """Run the agent against a scripted LLM. Returns (AgentResult, FakeLLM)."""
    llm = FakeLLM(turns)
    agent = GeofetchAgent(tools=tools or make_tools(), max_steps=max_steps)
    with mock.patch("core.pipeline.geofetch.chat_tools", llm):
        res = agent.run(PORTAL, "Atlantis land cover, forest layer", fmt, "latest")
    return res, llm


class TestAgentLoop(unittest.TestCase):
    def test_happy_path_resolves_and_verifies(self):
        res, _ = run_agent(list(HAPPY_PATH))
        self.assertTrue(res.found)
        self.assertEqual(res.download_url, FILE_URL)
        self.assertEqual(res.verification["payload_type"], "parquet")
        self.assertEqual(res.verification["total_size_bytes"], FILE_SIZE)
        self.assertEqual([t["tool"] for t in res.transcript],
                         ["fetch_page", "fetch_page", "fetch_page",
                          "probe_url", "report_result"])

    def test_token_usage_is_accumulated(self):
        res, llm = run_agent(list(HAPPY_PATH))
        self.assertEqual(res.steps_used, llm.chat_calls)  # one completion per step
        self.assertEqual(res.usage.total_tokens, 120 * res.steps_used)

    def test_tool_results_are_fed_back_to_model(self):
        _, llm = run_agent(list(HAPPY_PATH))
        tool_msgs = [m for m in llm.seen_messages if m.get("role") == "tool"]
        self.assertTrue(any("looks_like_js_app_shell" in m["content"] for m in tool_msgs))
        self.assertTrue(any("2025-07-01" in m["content"] for m in tool_msgs))

    def test_guardrail_rejects_unverifiable_report_then_recovers(self):
        dead = "https://api.atlantis.example/dl/download/GONE/forest.parquet"
        turns = [
            tool_turn("report_result", {**GOOD_REPORT, "download_url": dead}),
            tool_turn("probe_url", {"url": FILE_URL}, "Rejected - probing the other one."),
            tool_turn("report_result", GOOD_REPORT),
        ]
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        self.assertEqual(res.download_url, FILE_URL)
        rejections = [m for m in llm.seen_messages
                      if m.get("role") == "tool" and "REPORT REJECTED" in m.get("content", "")]
        self.assertEqual(len(rejections), 1)

    def test_premature_failure_rejected_then_accepted(self):
        """A give-up after too few HTTP requests is bounced back once with technique
        suggestions; a repeated failure report is then accepted."""
        failure = {"found": False, "failure_reason": "no GeoParquet distribution"}
        turns = [tool_turn("report_result", dict(failure)),
                 tool_turn("report_result", dict(failure))]
        res, llm = run_agent(turns)
        self.assertFalse(res.found)
        self.assertEqual(res.report["failure_reason"], "no GeoParquet distribution")
        nudges = [m for m in llm.seen_messages
                  if m.get("role") == "tool" and "FAILURE REJECTED" in m.get("content", "")]
        self.assertEqual(len(nudges), 1)
        self.assertIn("web_search", nudges[0]["content"])

    def test_failure_after_real_effort_is_accepted_immediately(self):
        tools = make_tools()
        tools.requests_made = 12  # simulate prior effort
        turns = [tool_turn("report_result", {"found": False, "failure_reason": "exhausted"})]
        res, llm = run_agent(turns, tools=tools)
        self.assertFalse(res.found)
        self.assertFalse(any("FAILURE REJECTED" in m.get("content", "")
                             for m in llm.seen_messages if m.get("role") == "tool"))

    def test_content_only_turn_gets_nudged(self):
        turns = [_Msg(content="Thinking out loud, no tool call...")] + list(HAPPY_PATH)
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        self.assertTrue(any("Continue using tools" in m.get("content", "")
                            for m in llm.seen_messages if m.get("role") == "user"))

    def test_undiscovered_url_rejected_as_invented(self):
        """Reporting a URL that never appeared in any tool result is blocked even if that
        URL is live - provenance is required."""
        turns = [tool_turn("report_result", dict(GOOD_REPORT)),  # no discovery yet
                 tool_turn("fetch_page", {"url": ED_2025}),
                 tool_turn("report_result", dict(GOOD_REPORT))]
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        invented = [m for m in llm.seen_messages
                    if m.get("role") == "tool" and "invented" in m.get("content", "")]
        self.assertEqual(len(invented), 1)

    def test_wrong_format_file_rejected(self):
        """A live parquet file is rejected when the task asked for Shapefile: liveness
        alone is not relevance."""
        failure = {"found": False, "failure_reason": "no shapefile found"}
        turns = list(HAPPY_PATH) + [tool_turn("report_result", dict(failure)),
                                    tool_turn("report_result", dict(failure))]
        res, llm = run_agent(turns, fmt="Shapefile")
        self.assertFalse(res.found)
        self.assertTrue(any("wrong dataset" in m.get("content", "")
                            for m in llm.seen_messages if m.get("role") == "tool"))

    def test_history_compaction_trims_old_tool_results(self):
        long = "x" * 1000
        messages = [{"role": "system", "content": long}, {"role": "user", "content": long}]
        for i in range(8):
            messages.append({"role": "assistant", "content": long, "tool_calls": []})
            messages.append({"role": "tool", "tool_call_id": str(i), "content": long})
        _compact_history(messages)
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertTrue(all("trimmed" in m["content"] for m in tool_msgs[:3]))
        self.assertTrue(all(m["content"] == long for m in tool_msgs[3:]))
        self.assertEqual(messages[0]["content"], long)  # system untouched
        self.assertEqual(messages[1]["content"], long)  # task untouched
        self.assertTrue(all("trimmed" in m["content"]
                            for m in messages if m["role"] == "assistant"))

    def test_nudge_contains_task_reminder(self):
        turns = [_Msg(content="rambling...")] + list(HAPPY_PATH)
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        nudges = [m for m in llm.seen_messages
                  if m.get("role") == "user" and "REMINDER" in m.get("content", "")]
        self.assertTrue(nudges)
        self.assertIn("Atlantis land cover", nudges[0]["content"])

    def test_repeated_identical_call_suppressed_not_refetched(self):
        """The second identical fetch_page is answered from memory with a REPEATED CALL
        warning and costs no HTTP request; the agent can then continue to success."""
        turns = [tool_turn("fetch_page", {"url": PORTAL}),
                 tool_turn("fetch_page", {"url": PORTAL})] + list(HAPPY_PATH)[1:]
        tools = make_tools()
        res, llm = run_agent(turns, tools=tools)
        self.assertTrue(res.found)
        repeated = [m for m in llm.seen_messages
                    if m.get("role") == "tool" and "REPEATED CALL" in m.get("content", "")]
        self.assertEqual(len(repeated), 1)
        self.assertIn("task_reminder", repeated[0]["content"])
        # HTTP accounting: portal once, feed, edition, probe, verification probe = 5
        # (the repeat cost nothing)
        self.assertEqual(tools.requests_made, 5)

    def test_different_args_are_not_treated_as_repeats(self):
        turns = [tool_turn("fetch_page", {"url": PORTAL}),
                 tool_turn("fetch_page", {"url": PORTAL, "accept": "application/xml"}),
                 ] + list(HAPPY_PATH)[1:]
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        self.assertFalse(any("REPEATED CALL" in m.get("content", "")
                             for m in llm.seen_messages if m.get("role") == "tool"))

    def test_inline_tool_call_extraction_variants(self):
        js = '{"name": "fetch_page", "arguments": {"url": "https://x.example"}}'
        self.assertEqual(extract_inline_tool_call(js),
                         ("fetch_page", {"url": "https://x.example"}))
        self.assertEqual(
            extract_inline_tool_call(f"<think>hmm</think>Sure:\n```json\n{js}\n```"),
            ("fetch_page", {"url": "https://x.example"}))
        self.assertEqual(
            extract_inline_tool_call(
                '{"name": "probe_url", "arguments": "{\\"url\\": \\"https://y.example\\"}"}'),
            ("probe_url", {"url": "https://y.example"}))
        self.assertIsNone(extract_inline_tool_call("just prose, no JSON"))
        self.assertIsNone(extract_inline_tool_call('{"name": "unknown_tool", "arguments": {}}'))

    def test_coerce_malformed_report(self):
        out = coerce_report_args({"error": "nothing found"})
        self.assertFalse(out["found"])
        self.assertIn("nothing found", out["failure_reason"])
        self.assertTrue(coerce_report_args({"download_url": "https://x.example/f.parquet"})["found"])
        out3 = coerce_report_args({"url": "https://x.example/f.parquet"})
        self.assertTrue(out3["found"])
        self.assertEqual(out3["download_url"], "https://x.example/f.parquet")

    def test_report_with_url_field_alias_is_accepted(self):
        turns = list(HAPPY_PATH[:-1]) + [
            tool_turn("report_result", {"found": True, "url": FILE_URL})]
        res, _ = run_agent(turns)
        self.assertTrue(res.found)
        self.assertEqual(res.download_url, FILE_URL)

    def test_rescued_inline_fetch_call_is_executed(self):
        """A tool call written as plain text (template drift) is parsed, executed, and its
        result fed back; the run then succeeds."""
        inline = _Msg(content='{"name": "fetch_page", "arguments": '
                              f'{{"url": "{PORTAL}"}}}}')
        res, llm = run_agent([inline] + list(HAPPY_PATH)[1:])
        self.assertTrue(res.found)
        self.assertTrue(res.transcript[0].get("rescued"))
        rescue_msgs = [m for m in llm.seen_messages
                       if m.get("role") == "user"
                       and "Result of your fetch_page call" in m.get("content", "")]
        self.assertEqual(len(rescue_msgs), 1)

    def test_rescued_malformed_failure_report_ends_run_after_effort(self):
        tools = make_tools()
        tools.requests_made = 12  # effort already spent
        inline = _Msg(content='{"name": "report_result", "arguments": '
                              '{"error": "portal blocks programmatic access"}}')
        res, _ = run_agent([inline], tools=tools)
        self.assertFalse(res.found)
        self.assertIn("portal blocks", res.report["failure_reason"])

    def test_step_budget_exhaustion(self):
        res, _ = run_agent([_Msg(content="...")] * 3, max_steps=3)
        self.assertFalse(res.found)
        self.assertIn("budget", res.report["failure_reason"])

    def test_never_reports_invented_url_that_404s(self):
        """A fabricated URL is stopped by the guardrail even if the model insists: budget
        runs out rather than returning a bad URL."""
        dead = "https://api.atlantis.example/dl/download/FAKE/forest.parquet"
        turns = [tool_turn("report_result", {**GOOD_REPORT, "download_url": dead})] * 4
        res, _ = run_agent(turns, max_steps=4)
        self.assertFalse(res.found)
        self.assertEqual(res.download_url, "")


# --------------------------------------------------------------------------- #
# Pipeline seam: SearchAngle -> Candidate
# --------------------------------------------------------------------------- #

ANGLE = SearchAngle(
    description="Atlantis national mapping agency land cover",
    channel_hint="national_geoportal",
    rationale="the authoritative producer",
    dataset="Atlantis land cover, forest layer, whole country",
    format="GeoParquet",
    vintage="latest",
)

SEED_HITS = {"results": [{"title": "Atlantis Geoportal - LANDCOVER", "url": PORTAL},
                         {"title": "Atlantis DL service", "url": FEED}]}


def run_stage(turns, seed=None, angle=ANGLE):
    """Drive resolve_angle() with a scripted LLM and a scripted seed search."""
    llm = FakeLLM(turns)
    tools = make_tools()
    tools.web_search = lambda q: (SEED_HITS if seed is None else seed)
    with mock.patch("core.pipeline.geofetch.WebTools", lambda **kw: tools), \
         mock.patch("core.pipeline.geofetch.chat_tools", llm):
        return resolve_angle("Atlantis", angle, log=lambda m: None)


class TestResolveAngle(unittest.TestCase):
    def test_verified_result_becomes_a_candidate(self):
        found, missed = run_stage(list(HAPPY_PATH))
        self.assertIsNone(missed)
        self.assertIsInstance(found, Candidate)
        self.assertEqual(found.url, PORTAL)  # the page we cite
        self.assertEqual(found.resource_url, FILE_URL)  # the endpoint that serves data
        self.assertEqual(found.verification["payload_type"], "parquet")
        self.assertEqual(found.verification["first_bytes_hex"][:8], "50415231")  # PAR1
        self.assertEqual(found.source, "geofetch")
        self.assertEqual(found.confidence, config.CONFIDENCE_BY_REPORT["high"])
        self.assertEqual(found.title, "Atlantis Geoportal - LANDCOVER")  # page, not edition

    def test_claim_carries_the_models_own_account_structurally(self):
        found, _ = run_stage(list(HAPPY_PATH))
        claim = found.claim
        self.assertEqual(claim["edition"], "LANDCOVER_PARQUET_ATL_2025-07-01")
        self.assertEqual(claim["vintage_date"], "2025-07-01")
        self.assertEqual(claim["file_size_bytes"], FILE_SIZE)
        # the raw band survives alongside the mapped float
        self.assertEqual(claim["confidence"], "high")
        self.assertEqual(found.confidence, config.CONFIDENCE_BY_REPORT["high"])

    def test_evidence_stays_a_list_not_joined_prose(self):
        """The audit trail is structured data; flattening it to a string was the loss this
        whole shape exists to undo."""
        found, _ = run_stage(list(HAPPY_PATH))
        self.assertIsInstance(found.claim["evidence"], list)
        self.assertEqual(found.claim["evidence"], [PORTAL, FEED, ED_2025, FILE_URL])

    def test_string_evidence_is_normalized_to_a_list(self):
        """Shape is normalized, truth is not - otherwise a consumer iterating `evidence`
        walks a bare string one character at a time."""
        turns = list(HAPPY_PATH[:-1]) + [
            tool_turn("report_result", {**GOOD_REPORT, "evidence": "found it on page 2"})]
        found, _ = run_stage(turns)
        self.assertEqual(found.claim["evidence"], ["found it on page 2"])

    def test_claim_is_a_whitelist_not_a_passthrough(self):
        """report is model-authored, so an unknown key - especially one that reads like a
        verification result - must not reach the output."""
        turns = list(HAPPY_PATH[:-1]) + [tool_turn("report_result", {
            **GOOD_REPORT, "ok": True, "payload_type": "parquet", "injected": "nope"})]
        found, _ = run_stage(turns)
        self.assertNotIn("injected", found.claim)
        self.assertNotIn("ok", found.claim)
        self.assertNotIn("payload_type", found.claim)
        self.assertNotIn("download_url", found.claim)  # duplicate of resource_url
        self.assertNotIn("found", found.claim)  # implied by which list it lands in

    def test_cost_is_measured_and_internally_consistent(self):
        found, _ = run_stage(list(HAPPY_PATH))
        cost = found.cost
        self.assertEqual(cost["steps_used"], len(HAPPY_PATH))  # one completion per step
        self.assertEqual(cost["total_tokens"],
                         cost["prompt_tokens"] + cost["completion_tokens"])
        self.assertGreater(cost["http_requests"], 0)

    def test_confidence_never_falls_below_the_verified_floor(self):
        turns = list(HAPPY_PATH[:-1]) + [
            tool_turn("report_result", {**GOOD_REPORT, "confidence": "not-a-band"})]
        found, _ = run_stage(turns)
        self.assertEqual(found.confidence, config.CONFIDENCE_DEFAULT)
        self.assertGreaterEqual(found.confidence, 0.7)

    def test_failure_becomes_an_unresolved_record_not_a_candidate(self):
        failure = {"found": False, "failure_reason": "no GeoParquet distribution"}
        found, missed = run_stage([tool_turn("report_result", dict(failure)),
                                   tool_turn("report_result", dict(failure))])
        self.assertIsNone(found)
        self.assertIsInstance(missed, Candidate)
        self.assertIsNone(missed.resource_url)
        self.assertIsNone(missed.confidence)
        self.assertIsNone(missed.verification)  # nothing to verify on a miss
        self.assertIn("no GeoParquet distribution", missed.claim["failure_reason"])
        # a dead end that burned the budget is exactly what cost is for
        self.assertGreater(missed.cost["steps_used"], 0)
        self.assertIn("total_tokens", missed.cost)

    def test_empty_seed_search_yields_nothing_at_all(self):
        found, missed = run_stage(list(HAPPY_PATH), seed={"results": []})
        self.assertIsNone(found)
        self.assertIsNone(missed)

    def test_malformed_seed_hits_are_dropped_not_fatal(self):
        """A hit with no usable url must not take the whole angle down on line one."""
        seed = {"results": [{"title": "junk"}, "not-a-dict",
                            {"title": "ok", "url": PORTAL}]}
        found, _ = run_stage(list(HAPPY_PATH), seed=seed)
        self.assertIsNotNone(found)
        self.assertEqual(found.url, PORTAL)

    def test_seed_hits_are_shown_to_the_model(self):
        """The caller already paid for a search; its results go into the task message so
        the agent uses them as leads instead of re-running an equivalent query."""
        llm = FakeLLM(list(HAPPY_PATH))
        tools = make_tools()
        tools.web_search = lambda q: SEED_HITS
        with mock.patch("core.pipeline.geofetch.WebTools", lambda **kw: tools), \
             mock.patch("core.pipeline.geofetch.chat_tools", llm):
            resolve_angle("Atlantis", ANGLE, log=lambda m: None)
        task_msg = llm.seen_messages[1]["content"]
        self.assertIn(FEED, task_msg)
        self.assertIn("Atlantis DL service", task_msg)

    def test_seed_hits_are_pre_seeded_into_provenance(self):
        """A URL from the seed search may be reported without being re-fetched - it did
        appear in a tool result, just an earlier one."""
        turns = [tool_turn("fetch_page", {"url": ED_2025}),
                 tool_turn("report_result", {**GOOD_REPORT, "download_url": FILE_URL})]
        found, _ = run_stage(turns)
        self.assertIsNotNone(found)


def _cand(url, resource, conf=0.95):
    return Candidate(url=url, title="t", source="geofetch",
                     confidence=conf, resource_url=resource)


class TestEscalationGate(unittest.TestCase):
    """Two conditions only, both about whether the PLAN worked. Candidate count is
    deliberately not one of them."""

    def test_empty_result_escalates(self):
        escalate, reason = needs_escalation([], [])
        self.assertTrue(escalate)
        self.assertIn("no angle resolved", reason)

    def test_one_verified_candidate_is_enough(self):
        """The point of dropping the count check: a single independently verified download
        is a good outcome, not something worth an expensive critic call."""
        escalate, reason = needs_escalation([_cand("https://a.example/p", "https://a.example/f.parquet")], [])
        self.assertFalse(escalate)
        self.assertEqual(reason, "")

    def test_more_dead_ends_than_wins_escalates(self):
        one = _cand("https://a.example/p", "https://a.example/f.parquet")
        misses = [_cand("https://b.example/p", None, None),
                  _cand("https://c.example/p", None, None)]
        escalate, reason = needs_escalation([one], misses)
        self.assertTrue(escalate)
        self.assertIn("2 of 3", reason)

    def test_same_domain_candidates_do_not_escalate(self):
        """Proves the publisher/domain-diversity branch is gone: two files from one host
        is a normal result for a country whose data lives on one national portal."""
        pool = [_cand("https://portal.example/a", "https://portal.example/a.parquet"),
                _cand("https://portal.example/b", "https://portal.example/b.parquet")]
        self.assertEqual(needs_escalation(pool, []), (False, ""))

    def test_candidates_without_resource_url_escalate(self):
        """Invariant guard: geofetch cannot produce these, so it means a broken contract."""
        escalate, reason = needs_escalation([_cand("https://a.example/p", None)], [])
        self.assertTrue(escalate)
        self.assertIn("fetchable resource endpoint", reason)


class TestRank(unittest.TestCase):
    _cand = staticmethod(_cand)

    def test_sorts_by_confidence_and_caps(self):
        pool = [self._cand(f"https://a.example/{i}", f"https://a.example/f{i}.parquet",
                           0.1 * i) for i in range(10)]
        ranked = _rank(pool)
        self.assertEqual(len(ranked), config.MAX_FINAL_CANDIDATES)
        self.assertEqual([c.confidence for c in ranked],
                         sorted([c.confidence for c in ranked], reverse=True))

    def test_dedupes_on_resource_url_first_wins(self):
        carried = self._cand("https://page.example/a", FILE_URL, 0.95)
        refound = self._cand("https://page.example/b", FILE_URL + "?x=1", 0.8)
        ranked = _rank([carried, refound])
        self.assertEqual(len(ranked), 1)
        self.assertIs(ranked[0], carried)  # carried is passed first, so it wins

    def test_falls_back_to_url_when_no_resource(self):
        a = self._cand("https://page.example/a", None, 0.9)
        b = self._cand("https://page.example/a/", None, 0.5)
        self.assertEqual(len(_rank([a, b])), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
