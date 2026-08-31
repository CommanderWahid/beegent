#!/usr/bin/env python3
"""run.py's ranking and the discover() loop."""

import os
import unittest
from unittest import mock

from beegent import config, llm
from beegent.connectors import LLMConnector
from beegent.run import _rank
from beegent.schemas import Candidate, SearchAngle, TokenUsage

from tests.fixtures import FILE_URL, _cand


class TestCliLoggingReachesTheTerminal(unittest.TestCase):
    """run.py is the one module that is ALSO an entry point, so __name__ is a trap there."""

    # A subprocess, because the bug only exists when run.py IS "__main__"; importing it
    # (as every other test does) binds the correct logger name and hides the fault.
    SCRIPT = """
import runpy, sys
from beegent.connectors import CONNECTORS, LLMConnector
from beegent.schemas import TokenUsage

class Fake(LLMConnector):
    provider = "fake"
    DEFAULT_MODELS = {"planner": "p", "geofetch": "g", "critic": "c"}
    def chat_json(self, model, messages):
        if model == "p":
            return {"angles": []}, TokenUsage(1, 1)          # no angles, so no geofetch
        return {"decision": "needs_human_review", "note": "n"}, TokenUsage(1, 1)
    def chat_tools(self, model, messages, tools):
        raise AssertionError("no angles, so this must never be reached")
    def validate(self):
        return None                                           # so main() gets past the gate

CONNECTORS["fake"] = Fake
sys.argv = ["run.py", "--country", "X", "--use-case", "Y", "--backend", "fake", "--out", %r]
runpy.run_module("beegent.run", run_name="__main__")
"""

    def test_run_py_logs_when_executed_as_main(self):
        """Under `python -m`, __name__ is "__main__" - outside the "beegent" logger hierarchy."""
        import subprocess
        import sys

        proc = subprocess.run([sys.executable, "-c", self.SCRIPT % os.devnull],
                              capture_output=True, text=True, timeout=60)
        out = proc.stdout + proc.stderr
        self.assertIn("[input]", out, "run.py's own logging must reach the terminal")
        self.assertIn("[config]", out)
        self.assertIn("--- summary", out, "including the summary block")


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


class TestThrottledAngleReachesTheOutput(unittest.TestCase):
    """Before failing soft, an angle whose LLM call errored left no trace at all."""

    def test_discover_records_the_failed_angle(self):
        from beegent import run as runmod

        good = SearchAngle("good", "web_search", "r", dataset="d", format="GeoParquet")
        bad = SearchAngle("throttled", "web_search", "r", dataset="d", format="GeoPackage")

        def fake_resolve(angle):
            if angle.format == "GeoParquet":
                return _cand("https://ok.example/p", "https://ok.example/f.parquet"), None
            # what resolve_angle now returns when the agent aborts mid-loop
            return None, Candidate(
                url="https://slow.example/p", title="T", source="geofetch",
                claim={"failure_reason": "agent aborted: RateLimitError: 429"},
                cost={"steps_used": 3, "http_requests": 2, "prompt_tokens": 300,
                      "completion_tokens": 60, "total_tokens": 360})

        with mock.patch.object(runmod, "plan", lambda c, u, f=None: [good, bad]), \
             mock.patch.object(runmod, "resolve_angle", fake_resolve), \
             mock.patch.object(runmod, "run_critic",
                               lambda *a: {"decision": "needs_human_review", "note": "n"}):
            run = runmod.discover("Atlantis", "land cover")

        out = run.to_dict()
        self.assertEqual(len(out["candidates"]), 1)
        self.assertEqual(len(out["unresolved"]), 1, "the throttled angle vanished")
        self.assertIn("429", out["unresolved"][0]["claim"]["failure_reason"])
        # and its spend is still attributed in the run totals
        self.assertEqual(out["totals"]["total_tokens"], 360)
        self.assertEqual(out["totals"]["angles_run"], 2)


class TestPlannerAndCriticSpendIsCounted(unittest.TestCase):
    """run.totals summed only Candidate.cost, so planner and critic tokens were invisible."""

    PLAN = {"angles": [{"description": "d", "url": "https://ok.example/p",
                        "dataset": "d", "format": "GeoParquet"}]}

    def _install(self, critic_decision="replan"):
        plan, verdict = self.PLAN, {"decision": critic_decision, "note": "n"}

        class C(LLMConnector):
            provider = "fake"
            DEFAULT_MODELS = {"planner": "p", "geofetch": "g", "critic": "c"}
            def chat_json(self, model, messages):
                if model == "p":
                    return plan, TokenUsage(800, 200)      # 1,000
                return verdict, TokenUsage(300, 100)       # 400
            def chat_tools(self, model, messages, tools):
                raise AssertionError("geofetch is mocked at resolve_angle in this test")
            def validate(self):
                return None

        patcher = mock.patch.object(llm, "_connector", C())  # not enterContext: 3.11+
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(llm.reset_usage)

    @staticmethod
    def _verified():
        return Candidate(
            url="https://ok.example/p", title="T", source="geofetch", confidence=0.95,
            resource_url="https://ok.example/f.parquet",
            cost={"http_requests": 4, "prompt_tokens": 5_000,
                  "completion_tokens": 1_000, "total_tokens": 6_000}), None

    def test_planner_bucket_reaches_totals_on_the_early_ok_exit(self):
        """The gate passes, so discover() returns at its first exit - the common path."""
        from beegent import run as runmod

        self._install()
        with mock.patch.object(runmod, "resolve_angle", lambda a: self._verified()):
            run = runmod.discover("Atlantis", "land cover")

        t = run.to_dict()["totals"]
        self.assertEqual(t["by_role"]["planner"]["total_tokens"], 1_000)
        self.assertEqual(t["by_role"]["geofetch"]["total_tokens"], 6_000)
        self.assertNotIn("critic", t["by_role"], "the gate passed, so no critic call")
        self.assertEqual(t["total_tokens"], 7_000, "run-wide total = sum of the buckets")

    def test_cached_tokens_reach_totals_from_both_metering_routes(self):
        """chat_json meters per role, chat_tools rides on Candidate.cost - both must carry it."""
        from beegent import run as runmod

        self._install()
        cached = Candidate(
            url="https://ok.example/p", title="T", source="geofetch", confidence=0.95,
            resource_url="https://ok.example/f.parquet",
            cost={"http_requests": 4, "prompt_tokens": 5_000, "completion_tokens": 1_000,
                  "total_tokens": 6_000, "cached_tokens": 4_200})
        with mock.patch.object(runmod, "resolve_angle", lambda a: (cached, None)):
            run = runmod.discover("Atlantis", "land cover")

        t = run.to_dict()["totals"]
        self.assertEqual(t["by_role"]["geofetch"]["cached_tokens"], 4_200)
        self.assertEqual(t["cached_tokens"], 4_200, "the planner reported none of its own")
        self.assertEqual(t["total_tokens"], 7_000, "cached must not inflate the run total")

    def test_a_backend_reporting_no_cache_leaves_the_key_at_zero(self):
        """The key is always present, so a run can be read without knowing the backend."""
        from beegent import run as runmod

        self._install()
        with mock.patch.object(runmod, "resolve_angle", lambda a: self._verified()):
            run = runmod.discover("Atlantis", "land cover")
        self.assertEqual(run.to_dict()["totals"]["cached_tokens"], 0)

    def test_a_raising_critic_call_does_not_lose_the_run(self):
        """Every stage fails soft: a throttled critic must not discard work already paid for."""
        from beegent import run as runmod

        class Throttled(LLMConnector):
            provider = "fake"
            DEFAULT_MODELS = {"planner": "p", "geofetch": "g", "critic": "c"}
            def chat_json(self, model, messages):
                if model == "p":
                    return TestPlannerAndCriticSpendIsCounted.PLAN, TokenUsage(800, 200)
                raise RuntimeError("Rate limit exceeded")   # the critic call, past SDK retries
            def chat_tools(self, model, messages, tools):
                raise AssertionError("geofetch is mocked at resolve_angle in this test")
            def validate(self):
                return None

        dead = Candidate(url="https://x.example/p", title="T", source="geofetch",
                         confidence=0.7, resource_url="",
                         claim={"failure_reason": "nothing verifiable"},
                         cost={"http_requests": 11, "prompt_tokens": 25_000,
                               "completion_tokens": 1_190, "total_tokens": 26_190})
        patcher = mock.patch.object(llm, "_connector", Throttled())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(llm.reset_usage)
        with mock.patch.object(runmod, "resolve_angle", lambda a: (None, dead)):
            run = runmod.discover("Atlantis", "land cover")

        self.assertEqual(run.status, "needs_human_review")
        self.assertIn("RuntimeError", run.reason, "the reason must say the critic was throttled")
        out = run.to_dict()
        self.assertEqual(out["totals"]["by_role"]["geofetch"]["total_tokens"], 26_190,
                         "work already paid for must survive")
        self.assertEqual(len(out["unresolved"]), 1, "the dead end must not vanish")

    def test_critic_bucket_is_counted_when_the_gate_fires(self):
        from beegent import run as runmod

        self._install(critic_decision="needs_human_review")
        dead = Candidate(url="https://x.example/p", title="T", source="geofetch",
                         claim={"failure_reason": "nothing"},
                         cost={"prompt_tokens": 900, "completion_tokens": 100,
                               "total_tokens": 1_000})
        with mock.patch.object(runmod, "resolve_angle", lambda a: (None, dead)):
            run = runmod.discover("Atlantis", "land cover")

        t = run.to_dict()["totals"]
        self.assertEqual(run.status, "needs_human_review")
        self.assertEqual(t["by_role"]["critic"]["total_tokens"], 400)
        self.assertEqual(t["total_tokens"], 1_000 + 1_000 + 400)

    def test_geofetch_is_not_double_counted(self):
        """Geofetch rides on Candidate.cost and is kept out of the meter, so nothing doubles."""
        from beegent import run as runmod

        self._install()
        with mock.patch.object(runmod, "resolve_angle", lambda a: self._verified()):
            run = runmod.discover("Atlantis", "land cover")

        self.assertNotIn("geofetch", llm.usage_by_role())
        self.assertEqual(run.totals["by_role"]["geofetch"]["total_tokens"], 6_000)

    def test_meter_does_not_leak_between_runs(self):
        from beegent import run as runmod

        self._install()
        with mock.patch.object(runmod, "resolve_angle", lambda a: self._verified()):
            first = runmod.discover("Atlantis", "land cover")
            second = runmod.discover("Atlantis", "land cover")
        self.assertEqual(first.totals["by_role"]["planner"],
                         second.totals["by_role"]["planner"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
