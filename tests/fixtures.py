"""Shared fixtures: a synthetic portal for a FICTIONAL country.

Deliberately not any real portal - if anything real leaked into the harness, these tests
would be the thing that stops noticing. A plain module rather than a pytest conftest, so it
works under `unittest discover` and pytest alike.
"""

import json

from unittest import mock

from beegent.pipeline.geofetch import GeofetchAgent, resolve_angle
from beegent.schemas import Candidate, SearchAngle, TokenUsage
from beegent.web_tools import HttpResult, WebTools

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
    """Scripted stand-in for beegent.llm.chat_tools: pops one (message, usage) pair per call
    and records the message history it was handed."""

    def __init__(self, turns, raise_on=None, error=None):
        self.turns = list(turns)
        self.seen_messages = []
        self.chat_calls = 0
        # raise_on: 1-based call number that should blow up instead of returning a turn,
        # standing in for a rate limit or a timeout that outlived the SDK's own retries.
        self.raise_on = raise_on
        self.error = error or RuntimeError("boom")

    def __call__(self, model, messages, tools):
        self.seen_messages = messages
        self.chat_calls += 1
        if self.raise_on is not None and self.chat_calls >= self.raise_on:
            raise self.error
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


def _cand(url, resource, conf=0.95):
    return Candidate(url=url, title="t", source="geofetch",
                     confidence=conf, resource_url=resource)


ANGLE = SearchAngle(
    description="Atlantis national mapping agency land cover",
    channel_hint="national_geoportal",
    rationale="the authoritative producer",
    url=PORTAL,
    dataset="Atlantis land cover, forest layer, whole country",
    format="GeoParquet",
    vintage="latest",
)


def run_agent(turns, tools=None, max_steps=10, fmt="GeoParquet"):
    """Run the agent against a scripted LLM. Returns (AgentResult, FakeLLM)."""
    llm = FakeLLM(turns)
    agent = GeofetchAgent(tools=tools or make_tools(), max_steps=max_steps)
    with mock.patch("beegent.pipeline.geofetch.chat_tools", llm):
        res = agent.run(PORTAL, "Atlantis land cover, forest layer", fmt, "latest")
    return res, llm


def run_stage(turns, angle=ANGLE, raise_on=None, error=None):
    """Drive resolve_angle() with a scripted LLM. No search stub: the angle carries the URL."""
    llm = FakeLLM(turns, raise_on=raise_on, error=error)
    tools = make_tools()
    with mock.patch("beegent.pipeline.geofetch.WebTools", lambda **kw: tools), \
         mock.patch("beegent.pipeline.geofetch.chat_tools", llm):
        return resolve_angle(angle, log=lambda m: None)
