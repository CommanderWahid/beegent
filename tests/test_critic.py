#!/usr/bin/env python3
"""The escalation gate (deterministic) and what reaches the critic's prompt."""

from beegent.pipeline import critic as cr
from beegent.pipeline.critic import needs_escalation
from beegent.schemas import Candidate, TokenUsage

from tests.conftest import ANGLE, make_candidate

# --- the escalation gate: ONE condition, was anything verified at all? ---


def test_empty_result_escalates():
    escalate, reason = needs_escalation([])
    assert escalate
    assert "no angle resolved" in reason


def test_one_verified_candidate_is_enough():
    """One verified download is a good outcome, not worth an expensive critic call."""
    escalate, reason = needs_escalation(
        [make_candidate("https://a.example/p", "https://a.example/f.parquet")])
    assert not escalate
    assert reason == ""


def test_dead_end_count_is_not_a_gate_condition():
    """One verified download passes no matter how many other angles missed."""
    assert needs_escalation(
        [make_candidate("https://a.example/p", "https://a.example/f.parquet")]) == (False, "")


def test_same_domain_candidates_do_not_escalate():
    """Two files from one host is a normal result; the diversity branch is gone."""
    pool = [make_candidate("https://portal.example/a", "https://portal.example/a.parquet"),
            make_candidate("https://portal.example/b", "https://portal.example/b.parquet")]
    assert needs_escalation(pool) == (False, "")


def test_candidates_without_resource_url_escalate():
    """Invariant guard: geofetch cannot produce these, so it means a broken contract."""
    escalate, reason = needs_escalation([make_candidate("https://a.example/p", None)])
    assert escalate
    assert "fetchable resource endpoint" in reason


# --- what reaches the critic: `unresolved` carries the only real signal it has ---


def test_a_raising_call_fails_safe_instead_of_propagating(monkeypatch):
    """run_critic() never raises: the SDK's retries are already spent by the time it does."""
    def boom(model, messages):
        raise RuntimeError("Rate limit exceeded")

    monkeypatch.setattr(cr, "chat_json", boom)
    verdict = cr.run_critic("Atlantis", "land cover", [], [], [], "nothing verified")
    assert verdict["decision"] == "needs_human_review"
    assert "RuntimeError" in verdict["note"]
    assert "Rate limit" in verdict["note"], "say WHY, or it reads as a real escalation"


def test_dead_end_reasons_reach_the_critic_prompt(monkeypatch):
    seen = {}

    def fake_chat_json(model, messages):
        seen["user"] = messages[1]["content"]
        return ({"decision": "replan", "note": "try the bulk file server"},
                TokenUsage(40, 10))

    misses = [Candidate(url="https://slow.example/p", title="t", source="geofetch",
                        claim={"failure_reason": "portal is login-walled"})]
    monkeypatch.setattr(cr, "chat_json", fake_chat_json)
    out = cr.run_critic("Atlantis", "land cover", [ANGLE], [], misses, "nothing verified")
    assert out["decision"] == "replan"
    assert "https://slow.example/p" in seen["user"]
    assert "portal is login-walled" in seen["user"]
