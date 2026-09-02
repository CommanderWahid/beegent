#!/usr/bin/env python3
"""The agent loop and its five guardrails, against the synthetic Atlantis portal."""

import json

import httpx
import openai
import pytest

from beegent import config
from beegent.pipeline.geofetch import (
    GeofetchAgent,
    TOOL_SCHEMAS,
    _compact_history,
    _fit,
    coerce_report_args,
    extract_inline_tool_call,
    resolve_angle,
)
from beegent.schemas import Candidate, SearchAngle
from beegent.web_tools import HttpResult, WebTools

from tests.conftest import (ANGLE, API_ERROR_URL, DEFAULT_PAGES, ED_2025, EDITION_XML, FEED,
                            FILE_SIZE, FILE_URL, GOOD_REPORT, HAPPY_PATH, OAPIF_EMPTY,
                            OAPIF_ITEMS, PORTAL, WFS_CAPS_URL, FakeLLM, _Msg, build_tools,
                            tool_turn)

# --- the agent loop ---


def test_happy_path_resolves_and_verifies(run_agent):
    res, _ = run_agent(list(HAPPY_PATH))
    assert res.found
    assert res.download_url == FILE_URL
    assert res.verification["payload_type"] == "parquet"
    assert res.verification["total_size_bytes"] == FILE_SIZE
    assert [t["tool"] for t in res.transcript] == [
        "fetch_page", "fetch_page", "fetch_page", "probe_url", "report_result"]


def test_token_usage_is_accumulated(run_agent):
    res, llm = run_agent(list(HAPPY_PATH))
    assert res.steps_used == llm.chat_calls  # one completion per step
    assert res.usage.total_tokens == 120 * res.steps_used


def test_tool_results_are_fed_back_to_model(run_agent):
    _, llm = run_agent(list(HAPPY_PATH))
    tool_msgs = [m for m in llm.seen_messages if m.get("role") == "tool"]
    assert any("looks_like_js_app_shell" in m["content"] for m in tool_msgs)
    assert any("2025-07-01" in m["content"] for m in tool_msgs)


def test_guardrail_rejects_unverifiable_report_then_recovers(run_agent):
    dead = "https://api.atlantis.example/dl/download/GONE/forest.parquet"
    turns = [
        tool_turn("report_result", {**GOOD_REPORT, "download_url": dead}),
        tool_turn("probe_url", {"url": FILE_URL}, "Rejected - probing the other one."),
        tool_turn("report_result", GOOD_REPORT),
    ]
    res, llm = run_agent(turns)
    assert res.found
    assert res.download_url == FILE_URL
    rejections = [m for m in llm.seen_messages
                  if m.get("role") == "tool" and "REPORT REJECTED" in m.get("content", "")]
    assert len(rejections) == 1


def test_premature_failure_rejected_then_accepted(run_agent):
    """A premature give-up is bounced once; a repeated failure report is then accepted."""
    failure = {"found": False, "failure_reason": "no GeoParquet distribution"}
    turns = [tool_turn("report_result", dict(failure)),
             tool_turn("report_result", dict(failure))]
    res, llm = run_agent(turns)
    assert not res.found
    assert res.report["failure_reason"] == "no GeoParquet distribution"
    nudges = [m for m in llm.seen_messages
              if m.get("role") == "tool" and "FAILURE REJECTED" in m.get("content", "")]
    assert len(nudges) == 1
    assert "web_search" in nudges[0]["content"]


def test_failure_after_real_effort_is_accepted_immediately(run_agent, tools):
    tools.requests_made = 12  # simulate prior effort
    turns = [tool_turn("report_result", {"found": False, "failure_reason": "exhausted"})]
    res, llm = run_agent(turns, tools=tools)
    assert not res.found
    assert not any("FAILURE REJECTED" in m.get("content", "")
                   for m in llm.seen_messages if m.get("role") == "tool")


def test_content_only_turn_gets_nudged(run_agent):
    turns = [_Msg(content="Thinking out loud, no tool call...")] + list(HAPPY_PATH)
    res, llm = run_agent(turns)
    assert res.found
    assert any("Continue using tools" in m.get("content", "")
               for m in llm.seen_messages if m.get("role") == "user")


def test_undiscovered_url_rejected_as_invented(run_agent):
    """A URL that appeared in no tool result is rejected even when live."""
    turns = [tool_turn("report_result", dict(GOOD_REPORT)),  # no discovery yet
             tool_turn("fetch_page", {"url": ED_2025}),
             tool_turn("report_result", dict(GOOD_REPORT))]
    res, llm = run_agent(turns)
    assert res.found
    invented = [m for m in llm.seen_messages
                if m.get("role") == "tool" and "invented" in m.get("content", "")]
    assert len(invented) == 1


def test_wrong_format_file_rejected(run_agent):
    """A live parquet is rejected when Shapefile was asked for."""
    failure = {"found": False, "failure_reason": "no shapefile found"}
    turns = list(HAPPY_PATH) + [tool_turn("report_result", dict(failure)),
                                tool_turn("report_result", dict(failure))]
    res, llm = run_agent(turns, fmt="Shapefile")
    assert not res.found
    assert any("wrong dataset" in m.get("content", "")
               for m in llm.seen_messages if m.get("role") == "tool")


def test_history_compaction_trims_old_tool_results():
    long = "x" * 1000
    messages = [{"role": "system", "content": long}, {"role": "user", "content": long}]
    for i in range(8):
        messages.append({"role": "assistant", "content": long, "tool_calls": []})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": long})
    _compact_history(messages)
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    keep = config.KEEP_FULL_TOOL_RESULTS  # derive, so retuning config cannot break this
    assert all("trimmed" in m["content"] for m in tool_msgs[:-keep])
    assert all(m["content"] == long for m in tool_msgs[-keep:])
    assert messages[0]["content"] == long  # system untouched
    assert messages[1]["content"] == long  # task untouched
    assert all("trimmed" in m["content"] for m in messages if m["role"] == "assistant")


def test_nudge_contains_task_reminder(run_agent):
    turns = [_Msg(content="rambling...")] + list(HAPPY_PATH)
    res, llm = run_agent(turns)
    assert res.found
    nudges = [m for m in llm.seen_messages
              if m.get("role") == "user" and "REMINDER" in m.get("content", "")]
    assert nudges
    assert "Atlantis land cover" in nudges[0]["content"]


def test_repeated_identical_call_suppressed_not_refetched(run_agent, tools):
    """A repeated call is answered from memory and costs no HTTP request."""
    turns = [tool_turn("fetch_page", {"url": PORTAL}),
             tool_turn("fetch_page", {"url": PORTAL})] + list(HAPPY_PATH)[1:]
    res, llm = run_agent(turns, tools=tools)
    assert res.found
    repeated = [m for m in llm.seen_messages
                if m.get("role") == "tool" and "REPEATED CALL" in m.get("content", "")]
    assert len(repeated) == 1
    assert "task_reminder" in repeated[0]["content"]
    # portal, feed, edition, probe, verification probe = 5; the repeat cost nothing
    assert tools.requests_made == 5


def test_repeats_abort_the_angle_once_it_is_clearly_stuck(run_agent):
    """The anti-loop guard used to warn forever."""
    turns = [tool_turn("fetch_page", {"url": PORTAL})] + [
        tool_turn("fetch_page", {"url": PORTAL}) for _ in range(6)]
    res, _ = run_agent(turns, max_steps=10)
    assert not res.found
    assert "not converging" in res.report["failure_reason"]
    assert str(config.GEOFETCH_MAX_REPEATS) in res.report["failure_reason"]
    # stopped early rather than burning the full budget
    assert res.steps_used < 10
    assert res.usage.total_tokens > 0, "partial cost still reported"


def test_the_guard_still_allows_recovery(run_agent):
    """Fewer repeats than the threshold must not end the angle."""
    assert config.GEOFETCH_MAX_REPEATS > 1
    turns = [tool_turn("fetch_page", {"url": PORTAL}),
             tool_turn("fetch_page", {"url": PORTAL})] + list(HAPPY_PATH)[1:]
    res, _ = run_agent(turns)
    assert res.found, "one suppression must not abort the angle"


def test_different_args_are_not_treated_as_repeats(run_agent):
    turns = [tool_turn("fetch_page", {"url": PORTAL}),
             tool_turn("fetch_page", {"url": PORTAL, "accept": "application/xml"}),
             ] + list(HAPPY_PATH)[1:]
    res, llm = run_agent(turns)
    assert res.found
    assert not any("REPEATED CALL" in m.get("content", "")
                   for m in llm.seen_messages if m.get("role") == "tool")


def test_inline_tool_call_extraction_variants():
    js = '{"name": "fetch_page", "arguments": {"url": "https://x.example"}}'
    assert extract_inline_tool_call(js) == ("fetch_page", {"url": "https://x.example"})
    assert extract_inline_tool_call(
        f"<think>hmm</think>Sure:\n```json\n{js}\n```") == (
        "fetch_page", {"url": "https://x.example"})
    assert extract_inline_tool_call(
        '{"name": "probe_url", "arguments": "{\\"url\\": \\"https://y.example\\"}"}') == (
        "probe_url", {"url": "https://y.example"})
    assert extract_inline_tool_call("just prose, no JSON") is None
    assert extract_inline_tool_call('{"name": "unknown_tool", "arguments": {}}') is None


def test_coerce_malformed_report():
    out = coerce_report_args({"error": "nothing found"})
    assert not out["found"]
    assert "nothing found" in out["failure_reason"]
    assert coerce_report_args({"download_url": "https://x.example/f.parquet"})["found"]
    out3 = coerce_report_args({"url": "https://x.example/f.parquet"})
    assert out3["found"]
    assert out3["download_url"] == "https://x.example/f.parquet"


def test_report_with_url_field_alias_is_accepted(run_agent):
    turns = list(HAPPY_PATH[:-1]) + [
        tool_turn("report_result", {"found": True, "url": FILE_URL})]
    res, _ = run_agent(turns)
    assert res.found
    assert res.download_url == FILE_URL


def test_rescued_inline_fetch_call_is_executed(run_agent):
    """A tool call written as plain text is parsed, executed and fed back."""
    inline = _Msg(content='{"name": "fetch_page", "arguments": '
                          f'{{"url": "{PORTAL}"}}}}')
    res, llm = run_agent([inline] + list(HAPPY_PATH)[1:])
    assert res.found
    assert res.transcript[0].get("rescued")
    rescue_msgs = [m for m in llm.seen_messages
                   if m.get("role") == "user"
                   and "Result of your fetch_page call" in m.get("content", "")]
    assert len(rescue_msgs) == 1


def test_repeated_inline_call_hits_the_anti_loop_guard(run_agent):
    """The rescue path used to skip _call_seen and loop to the step budget."""
    inline = _Msg(content='{"name": "fetch_page", "arguments": '
                          f'{{"url": "{PORTAL}"}}}}')
    res, _ = run_agent([inline] * 8, max_steps=8)
    assert not res.found
    assert "not converging" in res.report["failure_reason"]
    assert res.steps_used < 8, "must abort before the step budget"
    assert res.steps_used == config.GEOFETCH_MAX_REPEATS + 1


def test_rescued_malformed_failure_report_ends_run_after_effort(run_agent, tools):
    tools.requests_made = 12  # effort already spent
    inline = _Msg(content='{"name": "report_result", "arguments": '
                          '{"error": "portal blocks programmatic access"}}')
    res, _ = run_agent([inline], tools=tools)
    assert not res.found
    assert "portal blocks" in res.report["failure_reason"]


def test_step_budget_exhaustion(run_agent):
    res, _ = run_agent([_Msg(content="...")] * 3, max_steps=3)
    assert not res.found
    assert "budget" in res.report["failure_reason"]


def test_never_reports_invented_url_that_404s(run_agent):
    """A fabricated URL is refused even when the model insists."""
    dead = "https://api.atlantis.example/dl/download/FAKE/forest.parquet"
    turns = [tool_turn("report_result", {**GOOD_REPORT, "download_url": dead})] * 4
    res, _ = run_agent(turns, max_steps=4)
    assert not res.found
    assert res.download_url == ""


# --- pipeline seam: SearchAngle -> Candidate ---


def test_verified_result_becomes_a_candidate(run_stage):
    found, missed = run_stage(list(HAPPY_PATH))
    assert missed is None
    assert isinstance(found, Candidate)
    assert found.url == PORTAL  # the page we cite
    assert found.resource_url == FILE_URL  # the endpoint that serves data
    assert found.verification["payload_type"] == "parquet"
    assert found.verification["first_bytes_hex"][:8] == "50415231"  # PAR1
    assert found.source == "geofetch"
    assert found.confidence == config.CONFIDENCE_BY_REPORT["high"]
    assert found.title == ANGLE.description  # a label, not the edition
    assert found.title != found.url, "title used to duplicate url"


def test_claim_carries_the_models_own_account_structurally(run_stage):
    found, _ = run_stage(list(HAPPY_PATH))
    claim = found.claim
    assert claim["edition"] == "LANDCOVER_PARQUET_ATL_2025-07-01"
    assert claim["vintage_date"] == "2025-07-01"
    assert claim["file_size_bytes"] == FILE_SIZE
    # the raw band survives alongside the mapped float
    assert claim["confidence"] == "high"
    assert found.confidence == config.CONFIDENCE_BY_REPORT["high"]


def test_evidence_stays_a_list_not_joined_prose(run_stage):
    """The audit trail stays structured data rather than a flattened string."""
    found, _ = run_stage(list(HAPPY_PATH))
    assert isinstance(found.claim["evidence"], list)
    assert found.claim["evidence"] == [PORTAL, FEED, ED_2025, FILE_URL]


def test_string_evidence_is_normalized_to_a_list(run_stage):
    """Shape is normalized, truth is not."""
    turns = list(HAPPY_PATH[:-1]) + [
        tool_turn("report_result", {**GOOD_REPORT, "evidence": "found it on page 2"})]
    found, _ = run_stage(turns)
    assert found.claim["evidence"] == ["found it on page 2"]


def test_a_report_with_null_optionals_still_verifies(run_stage):
    """Groq validates arguments against our schema; a null for "no value" must be legal."""
    turns = list(HAPPY_PATH[:-1]) + [tool_turn("report_result", {
        **GOOD_REPORT, "checksum": None, "failure_reason": None})]
    found, _ = run_stage(turns)
    assert found is not None, "a null optional must not cost a verified download"
    assert found.verification["payload_type"] == "parquet"
    for key in ("checksum", "failure_reason"):
        assert key not in found.claim, "a null carries no claim"


def test_structured_evidence_survives_as_the_model_wrote_it(run_stage):
    """claim is what the model SAID - url+note beats a sentence, so do not flatten it."""
    chain = [{"step": "Fetched start page", "url": PORTAL, "note": "directory listing"},
             {"step": "Identified file", "url": FILE_URL, "note": "21.44 GB"}]
    turns = list(HAPPY_PATH[:-1]) + [
        tool_turn("report_result", {**GOOD_REPORT, "evidence": chain})]
    found, _ = run_stage(turns)
    assert found.claim["evidence"] == chain


def test_claim_is_a_whitelist_not_a_passthrough(run_stage):
    """An unknown model-authored key must not reach the output."""
    turns = list(HAPPY_PATH[:-1]) + [tool_turn("report_result", {
        **GOOD_REPORT, "ok": True, "payload_type": "parquet", "injected": "nope"})]
    found, _ = run_stage(turns)
    assert "injected" not in found.claim
    assert "ok" not in found.claim
    assert "payload_type" not in found.claim
    assert "download_url" not in found.claim  # duplicate of resource_url
    assert "found" not in found.claim  # implied by which list it lands in


def test_cost_is_measured_and_internally_consistent(run_stage):
    found, _ = run_stage(list(HAPPY_PATH))
    cost = found.cost
    assert cost["steps_used"] == len(HAPPY_PATH)  # one completion per step
    assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]
    assert cost["http_requests"] > 0


def test_confidence_never_falls_below_the_verified_floor(run_stage):
    turns = list(HAPPY_PATH[:-1]) + [
        tool_turn("report_result", {**GOOD_REPORT, "confidence": "not-a-band"})]
    found, _ = run_stage(turns)
    assert found.confidence == config.CONFIDENCE_DEFAULT
    assert found.confidence >= 0.7


def test_failure_becomes_an_unresolved_record_not_a_candidate(run_stage):
    failure = {"found": False, "failure_reason": "no GeoParquet distribution"}
    found, missed = run_stage([tool_turn("report_result", dict(failure)),
                               tool_turn("report_result", dict(failure))])
    assert found is None
    assert isinstance(missed, Candidate)
    assert missed.resource_url is None
    assert missed.confidence is None
    assert missed.verification is None  # nothing to verify on a miss
    assert "no GeoParquet distribution" in missed.claim["failure_reason"]
    assert missed.title == ANGLE.description  # dead ends get a label too
    # a dead end that burned the budget is exactly what cost is for
    assert missed.cost["steps_used"] > 0
    assert "total_tokens" in missed.cost


TRANSPORT_ERRORS = [
    openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
    openai.RateLimitError(
        "429 rate limit exceeded",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None),
]


@pytest.mark.parametrize("err", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__)
def test_llm_transport_error_becomes_an_unresolved_record(run_stage, err):
    """A rate limit or timeout that outlives the SDK's retries must not vaporize the angle."""
    found, missed = run_stage(list(HAPPY_PATH), raise_on=3, error=err)
    assert found is None
    assert missed is not None, "the angle vanished instead of failing soft"
    assert "agent aborted" in missed.claim["failure_reason"]
    assert type(err).__name__ in missed.claim["failure_reason"]
    # it must still report what it spent before dying
    assert missed.cost["steps_used"] == 3
    assert missed.cost["total_tokens"] > 0
    assert missed.cost["http_requests"] > 0


def test_transport_error_on_the_very_first_call_still_records(run_stage):
    found, missed = run_stage([], raise_on=1, error=RuntimeError("connection refused"))
    assert found is None
    assert "connection refused" in missed.claim["failure_reason"]
    assert missed.cost["steps_used"] == 1


def test_the_angles_url_is_what_the_agent_is_started_with(monkeypatch):
    """The planner's four values reach the agent verbatim - nothing is rediscovered."""
    llm = FakeLLM(list(HAPPY_PATH))
    tools = build_tools()
    monkeypatch.setattr("beegent.pipeline.geofetch.WebTools", lambda **kw: tools)
    monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", llm)
    resolve_angle(ANGLE, log=lambda m: None)
    task = json.loads(llm.seen_messages[1]["content"].split("\n", 1)[1])
    assert task == {"start_url": ANGLE.url, "dataset": ANGLE.dataset,
                    "format": ANGLE.format, "vintage": ANGLE.vintage}


def test_no_search_is_spent_before_the_agent_runs(monkeypatch, run_stage):
    """The pre-flight seed search is gone; a blocked engine cannot kill an angle."""
    llm = FakeLLM(list(HAPPY_PATH))
    tools = build_tools()
    calls = []
    tools.web_search = lambda q: calls.append(q) or {"results": []}
    monkeypatch.setattr("beegent.pipeline.geofetch.WebTools", lambda **kw: tools)
    monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", llm)
    found, _ = run_stage(list(HAPPY_PATH))
    assert calls == [], "resolve_angle still ran a pre-flight search"
    assert "web_search" in [t["function"]["name"] for t in TOOL_SCHEMAS]


# --- format containers: bulk geodata ships inside them, e.g. split .7z archives ---

HEADS = {"7z": b"7z\xbc\xaf\x27\x1c" + b"\x00" * 10,
         "zip": b"PK\x03\x04" + b"\x00" * 12,
         "gzip": b"\x1f\x8b" + b"\x00" * 14,
         "parquet": b"PAR1" + b"\x00" * 12}


def _verdict(fmt, kind):
    body = HEADS[kind]

    def transport(method, url, headers, max_bytes):
        return HttpResult(206, {"content-type": "application/octet-stream",
                                "content-range": "bytes 0-15/999"}, body[:16], url)
    a = GeofetchAgent(tools=WebTools(transport=transport))
    a._task_format, a._task_json = fmt, "{}"
    a._discovered = {"https://x.example/f"}
    return a._handle_report({"found": True, "download_url": "https://x.example/f"})


@pytest.mark.parametrize("kind", ("7z", "zip", "gzip"))
def test_geopackage_accepts_archive_containers(kind):
    assert _verdict("GeoPackage", kind) is not None


@pytest.mark.parametrize("kind,accepted", [("parquet", True), ("7z", False),
                                           ("zip", False), ("gzip", False)])
def test_geoparquet_stays_strict(kind, accepted):
    """A container must not satisfy the format most often asked for."""
    assert (_verdict("GeoParquet", kind) is not None) is accepted


# --- the tool schemas are a contract with a validator we do not control ---

FUNCTIONS = [f["function"] for f in TOOL_SCHEMAS]
OPTIONAL_PROPS = [(fn["name"], name, spec)
                  for fn in FUNCTIONS
                  for name, spec in fn["parameters"]["properties"].items()
                  if name not in fn["parameters"].get("required", [])]


@pytest.mark.parametrize("tool,name,spec", OPTIONAL_PROPS,
                         ids=[f"{t}.{n}" for t, n, _ in OPTIONAL_PROPS])
def test_optional_properties_carry_no_type_constraint(tool, name, spec):
    """Two verified downloads were binned for shapes we failed to anticipate; stop guessing."""
    assert "type" not in spec, "an unanticipated shape must not 400"
    assert "enum" not in spec, "enum is checked independently of type"
    assert spec.get("description"), "still steer the model with prose"


def test_required_properties_stay_typed():
    """A missing required argument is a real error, and our own dispatch reports it."""
    typed = {(fn["name"], name): spec.get("type")
             for fn in FUNCTIONS
             for name, spec in fn["parameters"]["properties"].items()
             if name in fn["parameters"].get("required", [])}
    assert typed == {("fetch_page", "url"): "string",
                     ("web_search", "query"): "string",
                     ("probe_url", "url"): "string",
                     ("report_result", "found"): "boolean"}


def test_the_schemas_serialize_to_json():
    """They go on the wire verbatim on every step, so a non-serializable value is fatal."""
    assert "report_result" in json.dumps(TOOL_SCHEMAS)


# --- every kept tool result is resent on every step, so an uncapped one is a per-step tax ---


def _page(n_urls=40, body_chars=None):
    """A fetch_page-shaped result whose body dwarfs everything else."""
    body = "z" * (body_chars or config.TOOL_RESULT_MAX_CHARS * 3)
    return {"requested_url": PORTAL, "final_url": PORTAL, "status": 200,
            "content_type": "application/xml", "truncated": False, "body": body,
            "urls_found": [f"https://atlantis.example/f/{i}.parquet" for i in range(n_urls)]}


def test_oversized_result_is_capped_before_entering_history():
    out = _fit(_page())
    assert len(out) < config.TOOL_RESULT_MAX_CHARS + 200
    assert "truncated" in out  # the model must know it was cut


def test_small_result_is_untouched():
    small = {"url": "u", "ok": True}
    assert json.loads(_fit(small)) == small
    assert "truncated_fields" not in _fit(small)


def test_a_non_dict_result_is_still_capped():
    """_fit is called directly in tests; a stray non-dict must not escape the budget."""
    out = _fit("z" * (config.TOOL_RESULT_MAX_CHARS * 5))
    assert len(out) < config.TOOL_RESULT_MAX_CHARS + 200


def test_urls_found_survives_when_the_body_does_not():
    """The regression this exists to prevent: urls_found is what the agent navigates by."""
    page = _page()
    out = json.loads(_fit(page))
    assert out["urls_found"] == page["urls_found"], "every URL must survive"
    assert len(out["body"]) < len(page["body"]), "the body is what pays"
    assert out["truncated_fields"] == ["body"]


@pytest.mark.parametrize("cap", (400, 1_000, 4_000, 20_000))
def test_output_is_always_valid_json(monkeypatch, cap):
    """A JSON document cut mid-token is its own hazard for a weak model."""
    monkeypatch.setattr(config, "TOOL_RESULT_MAX_CHARS", cap)
    out = json.loads(_fit(_page()))  # raises if it is not valid JSON
    assert out["requested_url"] == PORTAL, "identity fields always survive"


def test_scalars_alone_over_cap_degrade_without_raising(monkeypatch):
    monkeypatch.setattr(config, "TOOL_RESULT_MAX_CHARS", 50)
    assert len(_fit(_page())) < 50 + 200


def test_web_search_drops_trailing_results_not_mid_entry():
    """Search hits are ranked, so the tail is the cheapest part to lose."""
    hits = [{"title": f"hit {i}", "href": f"https://atlantis.example/{i}",
             "snippet": "s" * 300} for i in range(60)]
    out = json.loads(_fit({"query": "q", "engine": "ddg-html", "results": hits}))
    assert len(out["results"]) > 0, "some hits must survive"
    assert len(out["results"]) < len(hits)
    assert out["results"] == hits[: len(out["results"])], "kept from the front"
    assert out["truncated_fields"] == ["results"]


def test_a_real_oversized_page_still_shows_the_agent_the_file_url(make_tools):
    """The whole seam: WebTools output -> _fit -> what the model actually reads."""
    padded = EDITION_XML.replace("</feed>", "<junk>" + "q" * 60_000 + "</junk></feed>")
    tools = make_tools(pages={**DEFAULT_PAGES, ED_2025: (200, "application/xml", padded)})
    page = tools.fetch_page(ED_2025)
    assert len(json.dumps(page)) > config.TOOL_RESULT_MAX_CHARS, "must be over cap"
    out = json.loads(_fit(page))
    assert FILE_URL in out["urls_found"], "the one URL that matters must reach the model"
    assert out["truncated_fields"] == ["body"]


def test_a_link_far_down_a_long_page_still_reaches_the_model(make_tools):
    """The realistic shape: cutting the serialized tail loses exactly this URL."""
    buried = EDITION_XML.replace("<entry>", "<junk>" + "q" * 12_000 + "</junk><entry>")
    tools = make_tools(pages={**DEFAULT_PAGES, ED_2025: (200, "application/xml", buried)})
    out = json.loads(_fit(tools.fetch_page(ED_2025)))
    assert FILE_URL in out["urls_found"]
    assert FILE_URL not in out["body"], "it is past the cut - urls_found is why it survives"


def test_cap_applies_in_the_agent_loop(monkeypatch, make_tools):
    pages = dict(DEFAULT_PAGES)
    pages[FEED] = (200, "application/atom+xml", "<feed>" + "q" * 80_000 + "</feed>")
    llm = FakeLLM([tool_turn("fetch_page", {"url": FEED})] + list(HAPPY_PATH)[1:])
    agent = GeofetchAgent(tools=make_tools(pages=pages), max_steps=6)
    monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", llm)
    agent.run(PORTAL, "d", "GeoParquet", "latest")
    biggest = max(len(m["content"]) for m in llm.seen_messages if m.get("role") == "tool")
    assert biggest < config.TOOL_RESULT_MAX_CHARS + 200


# --- feature services: a reply is not proof, so the harness parses and counts ---


def _report(url):
    return tool_turn("report_result", {"found": True, "download_url": url,
                                       "confidence": "high"})


def test_a_verified_api_endpoint_carries_the_measured_field_set(run_agent):
    turns = [tool_turn("probe_url", {"url": OAPIF_ITEMS}), _report(OAPIF_ITEMS)]
    res, _ = run_agent(turns, fmt="GeoJSON")
    assert res.found
    v = res.verification
    assert v["access"] == "api"
    assert v["shape"] == "geojson_featurecollection"
    assert v["feature_count"] == 4212
    assert v["count_is_exact"]
    assert v["geometry_type"] == "Polygon"
    assert v["total_size_bytes"] is None, "a page's length is not the dataset's size"


def test_zero_features_is_rejected_then_the_agent_recovers(run_agent):
    """Live, well-formed and empty is not the dataset - the API analogue of a format mismatch."""
    turns = [tool_turn("probe_url", {"url": OAPIF_EMPTY}),
             tool_turn("probe_url", {"url": OAPIF_ITEMS}),
             _report(OAPIF_EMPTY), _report(OAPIF_ITEMS)]
    res, llm = run_agent(turns, fmt="GeoJSON")
    assert res.found
    assert res.download_url == OAPIF_ITEMS
    rejections = [m for m in llm.seen_messages
                  if m.get("role") == "tool" and "ZERO" in m.get("content", "")]
    assert len(rejections) == 1, "the empty endpoint must be bounced exactly once"


def test_a_capabilities_document_is_rejected_as_not_data(run_agent):
    """GetCapabilities lists the layers; it is metadata, and it used to pass as XML."""
    turns = [tool_turn("probe_url", {"url": WFS_CAPS_URL})] + [_report(WFS_CAPS_URL)] * 3
    res, llm = run_agent(turns, fmt="GeoJSON", max_steps=4)
    assert not res.found
    assert any("capabilities document" in m.get("content", "")
               for m in llm.seen_messages if m.get("role") == "tool")


def test_an_error_reply_served_with_http_200_is_rejected(run_agent):
    """The exact case a leading '{' waved through before."""
    turns = [tool_turn("probe_url", {"url": API_ERROR_URL})] + [_report(API_ERROR_URL)] * 3
    res, llm = run_agent(turns, fmt="GeoJSON", max_steps=4)
    assert not res.found
    assert any("not a recognisable feature collection" in m.get("content", "")
               for m in llm.seen_messages if m.get("role") == "tool")


def test_an_api_candidate_reaches_the_pipeline_seam(run_stage):
    """resolve_angle maps the service fields through to Candidate.verification unchanged."""
    angle = SearchAngle(description="Atlantis land cover service", channel_hint="catalog",
                        rationale="r", url=PORTAL, dataset="land cover", format="GeoJSON")
    found, missed = run_stage([tool_turn("probe_url", {"url": OAPIF_ITEMS}),
                               _report(OAPIF_ITEMS)], angle=angle)
    assert missed is None
    assert found.resource_url == OAPIF_ITEMS
    assert found.verification["access"] == "api"
    assert found.verification["feature_count"] == 4212


def test_the_prompt_teaches_feature_service_conventions():
    """Much of the world publishes only a queryable service; the prompt must say so."""
    from beegent.pipeline.geofetch import SYSTEM_PROMPT

    for token in ("/collections/", "GetFeature", "FeatureServer", "geoserver"):
        assert token in SYSTEM_PROMPT, token
    assert "GetCapabilities only LISTS" in SYSTEM_PROMPT, "and that capabilities is not data"


def test_the_prompt_asks_for_native_script_search_without_naming_a_language():
    """Examples were French and Spanish - instances, and the exact regional skew to avoid."""
    from beegent.pipeline.geofetch import SYSTEM_PROMPT

    assert "own script" in SYSTEM_PROMPT
    for instance in ("telechargement", "descarga"):
        assert instance not in SYSTEM_PROMPT, f"{instance!r} teaches an instance, not a method"
