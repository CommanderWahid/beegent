#!/usr/bin/env python3
"""The agent loop and its five guardrails, driven by a scripted model against the
synthetic Atlantis portal."""

import unittest
from unittest import mock

import json

from beegent import config
from beegent.pipeline.geofetch import (
    GeofetchAgent,
    _compact_history,
    coerce_report_args,
    extract_inline_tool_call,
    resolve_angle,
)
from beegent.schemas import Candidate
from beegent.web_tools import HttpResult, WebTools

from tests.fixtures import (ANGLE, DEFAULT_PAGES, ED_2025, FEED, FILE_SIZE, FILE_URL,
                            GOOD_REPORT, HAPPY_PATH, PORTAL, FakeLLM, _Msg, make_tools,
                            run_agent, run_stage, tool_turn)


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
        keep = config.KEEP_FULL_TOOL_RESULTS  # derive, so retuning config cannot break this
        self.assertTrue(all("trimmed" in m["content"] for m in tool_msgs[:-keep]))
        self.assertTrue(all(m["content"] == long for m in tool_msgs[-keep:]))
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

    def test_repeats_abort_the_angle_once_it_is_clearly_stuck(self):
        """The anti-loop guard used to warn forever. An observed run hit 7 suppressions with
        zero attempts to finish, burning ~180k tokens for nothing - every further step
        re-sends the whole conversation."""
        turns = [tool_turn("fetch_page", {"url": PORTAL})] + [
            tool_turn("fetch_page", {"url": PORTAL}) for _ in range(6)]
        res, _ = run_agent(turns, max_steps=10)
        self.assertFalse(res.found)
        self.assertIn("not converging", res.report["failure_reason"])
        self.assertIn(str(config.GEOFETCH_MAX_REPEATS), res.report["failure_reason"])
        # stopped early rather than burning the full budget
        self.assertLess(res.steps_used, 10)
        self.assertGreater(res.usage.total_tokens, 0, "partial cost still reported")

    def test_the_guard_still_allows_recovery(self):
        """Fewer repeats than the threshold must NOT end the angle - a real run recovered at
        step 7 after one suppression by switching to web_search, which is exactly what the
        REPEATED CALL message asks for."""
        self.assertGreater(config.GEOFETCH_MAX_REPEATS, 1)
        turns = [tool_turn("fetch_page", {"url": PORTAL}),
                 tool_turn("fetch_page", {"url": PORTAL})] + list(HAPPY_PATH)[1:]
        res, _ = run_agent(turns)
        self.assertTrue(res.found, "one suppression must not abort the angle")

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
        self.assertEqual(found.title, ANGLE.url)  # the cited page, not the edition

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

    def test_llm_transport_error_becomes_an_unresolved_record(self):
        """A rate limit or timeout that outlives the SDK's retries must not vaporize the
        angle. The one most likely to be throttled is the one you most need to see."""
        import openai
        import httpx

        cases = [
            openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
            openai.RateLimitError(
                "429 rate limit exceeded",
                response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                body=None),
        ]
        for err in cases:
            with self.subTest(error=type(err).__name__):
                found, missed = run_stage(list(HAPPY_PATH), raise_on=3, error=err)
                self.assertIsNone(found)
                self.assertIsNotNone(missed, "the angle vanished instead of failing soft")
                self.assertIn("agent aborted", missed.claim["failure_reason"])
                self.assertIn(type(err).__name__, missed.claim["failure_reason"])
                # it must still report what it spent before dying
                self.assertEqual(missed.cost["steps_used"], 3)
                self.assertGreater(missed.cost["total_tokens"], 0)
                self.assertGreater(missed.cost["http_requests"], 0)

    def test_transport_error_on_the_very_first_call_still_records(self):
        found, missed = run_stage([], raise_on=1, error=RuntimeError("connection refused"))
        self.assertIsNone(found)
        self.assertIn("connection refused", missed.claim["failure_reason"])
        self.assertEqual(missed.cost["steps_used"], 1)

    def test_the_angles_url_is_what_the_agent_is_started_with(self):
        """The planner's four values reach the agent verbatim - nothing is rediscovered."""
        llm = FakeLLM(list(HAPPY_PATH))
        tools = make_tools()
        with mock.patch("beegent.pipeline.geofetch.WebTools", lambda **kw: tools), \
             mock.patch("beegent.pipeline.geofetch.chat_tools", llm):
            resolve_angle(ANGLE, log=lambda m: None)
        task = json.loads(llm.seen_messages[1]["content"].split("\n", 1)[1])
        self.assertEqual(task, {"start_url": ANGLE.url, "dataset": ANGLE.dataset,
                                "format": ANGLE.format, "vintage": ANGLE.vintage})

    def test_no_search_is_spent_before_the_agent_runs(self):
        """The pre-flight seed search is gone: a blocked engine can no longer kill an angle
        before it starts. web_search survives as a TOOL the agent may call itself."""
        from beegent.pipeline.geofetch import TOOL_SCHEMAS
        llm = FakeLLM(list(HAPPY_PATH))
        tools = make_tools()
        calls = []
        tools.web_search = lambda q: calls.append(q) or {"results": []}
        with mock.patch("beegent.pipeline.geofetch.WebTools", lambda **kw: tools), \
             mock.patch("beegent.pipeline.geofetch.chat_tools", llm):
            found, _ = run_stage(list(HAPPY_PATH))
        self.assertEqual(calls, [], "resolve_angle still ran a pre-flight search")
        self.assertIn("web_search", [t["function"]["name"] for t in TOOL_SCHEMAS])


class TestFormatContainers(unittest.TestCase):
    """Bulk geodata ships inside containers - IGN publishes national GeoPackage as split
    .7z.001 archives. `zip` was already accepted on that reasoning; 7z/gzip complete it."""

    HEADS = {"7z": b"7z\xbc\xaf\x27\x1c" + b"\x00" * 10,
             "zip": b"PK\x03\x04" + b"\x00" * 12,
             "gzip": b"\x1f\x8b" + b"\x00" * 14,
             "parquet": b"PAR1" + b"\x00" * 12}

    def _verdict(self, fmt, kind):
        body = self.HEADS[kind]

        def transport(method, url, headers, max_bytes):
            return HttpResult(206, {"content-type": "application/octet-stream",
                                    "content-range": "bytes 0-15/999"}, body[:16], url)
        a = GeofetchAgent(tools=WebTools(transport=transport))
        a._task_format, a._task_json = fmt, "{}"
        a._discovered = {"https://x.example/f"}
        return a._handle_report({"found": True, "download_url": "https://x.example/f"})

    def test_geopackage_accepts_archive_containers(self):
        for kind in ("7z", "zip", "gzip"):
            with self.subTest(container=kind):
                self.assertIsNotNone(self._verdict("GeoPackage", kind))

    def test_geoparquet_stays_strict(self):
        """The one place this guardrail earns its keep: a container must NOT satisfy the
        format that is asked for most often."""
        self.assertIsNotNone(self._verdict("GeoParquet", "parquet"))
        for kind in ("7z", "zip", "gzip"):
            with self.subTest(container=kind):
                self.assertIsNone(self._verdict("GeoParquet", kind))


class TestToolResultCap(unittest.TestCase):
    """Every kept tool result is resent on every step, so an uncapped one is a per-step tax.
    A 60KB WFS GetCapabilities once drove a single angle to 861k input tokens."""

    def test_oversized_result_is_capped_before_entering_history(self):
        from beegent.pipeline.geofetch import _fit
        huge = "z" * (config.TOOL_RESULT_MAX_CHARS * 5)
        out = _fit(huge)
        self.assertLess(len(out), config.TOOL_RESULT_MAX_CHARS + 200)
        self.assertIn("truncated", out)  # the model must know it was cut

    def test_small_result_is_untouched(self):
        from beegent.pipeline.geofetch import _fit
        self.assertEqual(_fit("small"), "small")

    def test_cap_applies_in_the_agent_loop(self):
        pages = dict(DEFAULT_PAGES)
        pages[FEED] = (200, "application/atom+xml", "<feed>" + "q" * 80_000 + "</feed>")
        llm = FakeLLM([tool_turn("fetch_page", {"url": FEED})] + list(HAPPY_PATH)[1:])
        agent = GeofetchAgent(tools=make_tools(pages=pages), max_steps=6)
        with mock.patch("beegent.pipeline.geofetch.chat_tools", llm):
            agent.run(PORTAL, "d", "GeoParquet", "latest")
        biggest = max(len(m["content"]) for m in llm.seen_messages if m.get("role") == "tool")
        self.assertLess(biggest, config.TOOL_RESULT_MAX_CHARS + 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
