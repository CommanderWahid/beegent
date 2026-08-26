#!/usr/bin/env python3
"""run.py's ranking and the discover() loop."""

import unittest
from unittest import mock

from beegent import config
from beegent.run import _rank
from beegent.schemas import Candidate, SearchAngle

from tests.fixtures import FILE_URL, _cand


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
    """The regression that motivated failing soft: before this, an angle whose LLM call
    errored left no trace at all in candidate_list.json - no unresolved entry, no cost."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
