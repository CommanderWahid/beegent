#!/usr/bin/env python3
"""The escalation gate (deterministic) and what reaches the critic's prompt."""

import unittest
from unittest import mock

from beegent.pipeline import critic as cr
from beegent.pipeline.critic import needs_escalation
from beegent.schemas import Candidate, TokenUsage

from tests.fixtures import ANGLE, _cand


class TestEscalationGate(unittest.TestCase):
    """ONE condition: was anything verified at all?"""

    def test_empty_result_escalates(self):
        escalate, reason = needs_escalation([])
        self.assertTrue(escalate)
        self.assertIn("no angle resolved", reason)

    def test_one_verified_candidate_is_enough(self):
        """One verified download is a good outcome, not worth an expensive critic call."""
        escalate, reason = needs_escalation(
            [_cand("https://a.example/p", "https://a.example/f.parquet")])
        self.assertFalse(escalate)
        self.assertEqual(reason, "")

    def test_dead_end_count_is_not_a_gate_condition(self):
        """One verified download passes no matter how many other angles missed."""
        self.assertEqual(
            needs_escalation([_cand("https://a.example/p", "https://a.example/f.parquet")]),
            (False, ""))

    def test_same_domain_candidates_do_not_escalate(self):
        """Two files from one host is a normal result; the diversity branch is gone."""
        pool = [_cand("https://portal.example/a", "https://portal.example/a.parquet"),
                _cand("https://portal.example/b", "https://portal.example/b.parquet")]
        self.assertEqual(needs_escalation(pool), (False, ""))

    def test_candidates_without_resource_url_escalate(self):
        """Invariant guard: geofetch cannot produce these, so it means a broken contract."""
        escalate, reason = needs_escalation([_cand("https://a.example/p", None)])
        self.assertTrue(escalate)
        self.assertIn("fetchable resource endpoint", reason)


class TestCriticInput(unittest.TestCase):
    """`unresolved` carries the only real signal the critic has."""

    def test_a_raising_call_fails_safe_instead_of_propagating(self):
        """run_critic() never raises: the SDK's retries are already spent by the time it does."""
        from beegent.pipeline import critic as mod

        with mock.patch.object(mod, "chat_json", side_effect=RuntimeError("Rate limit exceeded")):
            verdict = mod.run_critic("Atlantis", "land cover", [], [], [], "nothing verified")
        self.assertEqual(verdict["decision"], "needs_human_review")
        self.assertIn("RuntimeError", verdict["note"])
        self.assertIn("Rate limit", verdict["note"], "say WHY, or it reads as a real escalation")

    def test_dead_end_reasons_reach_the_critic_prompt(self):
        seen = {}

        def fake_chat_json(model, messages):
            seen["user"] = messages[1]["content"]
            return ({"decision": "replan", "note": "try the bulk file server"},
                    TokenUsage(40, 10))

        misses = [Candidate(url="https://slow.example/p", title="t", source="geofetch",
                            claim={"failure_reason": "portal is login-walled"})]
        with mock.patch.object(cr, "chat_json", fake_chat_json):
            out = cr.run_critic("Atlantis", "land cover", [ANGLE], [], misses, "nothing verified")
        self.assertEqual(out["decision"], "replan")
        self.assertIn("https://slow.example/p", seen["user"])
        self.assertIn("portal is login-walled", seen["user"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
