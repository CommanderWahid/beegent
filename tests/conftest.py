"""Shared test data and fixtures: a synthetic portal for a FICTIONAL country."""

import json
import os

import pytest

from beegent import llm
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
    """Build a transport closure over fake pages and Range-honouring binary files."""
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


def build_tools(pages=None, binaries=None):
    """WebTools over the synthetic portal - the make_tools fixture hands this out."""
    return WebTools(transport=fake_transport(pages or DEFAULT_PAGES,
                                             binaries or {FILE_URL: FILE_SIZE}))


# --- scripted LLM, shaped like the SDK message chat_tools returns ---

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
    """Scripted chat_tools: pops one (message, usage) pair per call, records the history."""

    def __init__(self, turns, raise_on=None, error=None):
        self.turns = list(turns)
        self.seen_messages = []
        self.chat_calls = 0
        # raise_on: 1-based call number that blows up instead of returning a turn.
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


def make_candidate(url, resource, conf=0.95):
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


# --- fixtures ---


@pytest.fixture
def make_tools():
    """Factory for a WebTools over custom pages/binaries."""
    return build_tools


@pytest.fixture
def tools():
    """WebTools over the default synthetic portal."""
    return build_tools()


@pytest.fixture
def run_agent(monkeypatch):
    """Run the agent against a scripted LLM. Returns (AgentResult, FakeLLM)."""
    def _run(turns, tools=None, max_steps=10, fmt="GeoParquet"):
        fake = FakeLLM(turns)
        agent = GeofetchAgent(tools=tools or build_tools(), max_steps=max_steps)
        monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", fake)
        return agent.run(PORTAL, "Atlantis land cover, forest layer", fmt, "latest"), fake
    return _run


@pytest.fixture
def run_stage(monkeypatch):
    """Drive resolve_angle() with a scripted LLM. No search stub: the angle carries the URL."""
    def _run(turns, angle=ANGLE, raise_on=None, error=None):
        fake = FakeLLM(turns, raise_on=raise_on, error=error)
        built = build_tools()
        monkeypatch.setattr("beegent.pipeline.geofetch.WebTools", lambda **kw: built)
        monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", fake)
        return resolve_angle(angle, log=lambda m: None)
    return _run


@pytest.fixture
def clean_env(monkeypatch):
    """An empty os.environ - monkeypatch has no clear-all of its own."""
    for key in list(os.environ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def reset_llm_usage():
    """The per-role meter is module-level, so a test that spends starts and ends clean."""
    llm.reset_usage()
    yield
    llm.reset_usage()
