#!/usr/bin/env python3
"""run.py's ranking and the discover() loop."""

import os

from beegent import config, llm
from beegent.connectors import LLMConnector
from beegent.run import _rank
from beegent.schemas import Candidate, DiscoveryRun, SearchAngle, TokenUsage

from tests.conftest import FILE_URL, make_candidate

# --- run.py is the one module that is ALSO an entry point, so __name__ is a trap there ---

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


def test_run_py_logs_when_executed_as_main():
    """Under `python -m`, __name__ is "__main__" - outside the "beegent" logger hierarchy."""
    import subprocess
    import sys

    # A fixture cannot reach into a subprocess, so the store is disabled through the env.
    proc = subprocess.run([sys.executable, "-c", SCRIPT % os.devnull],
                          capture_output=True, text=True, timeout=60,
                          env={**os.environ, "BEEGENT_DB": ""})
    out = proc.stdout + proc.stderr
    assert "[input]" in out, "run.py's own logging must reach the terminal"
    assert "[config]" in out
    assert "--- summary" in out, "including the summary block"


# --- _rank(): dedupe, sort, cap ---


def test_sorts_by_confidence_and_caps():
    pool = [make_candidate(f"https://a.example/{i}", f"https://a.example/f{i}.parquet",
                           0.1 * i) for i in range(10)]
    ranked = _rank(pool)
    assert len(ranked) == config.MAX_FINAL_CANDIDATES
    assert [c.confidence for c in ranked] == sorted(
        [c.confidence for c in ranked], reverse=True)


def test_dedupes_on_resource_url_first_wins():
    carried = make_candidate("https://page.example/a", FILE_URL, 0.95)
    refound = make_candidate("https://page.example/b", FILE_URL + "?x=1", 0.8)
    ranked = _rank([carried, refound])
    assert len(ranked) == 1
    assert ranked[0] is carried  # carried is passed first, so it wins


def test_falls_back_to_url_when_no_resource():
    a = make_candidate("https://page.example/a", None, 0.9)
    b = make_candidate("https://page.example/a/", None, 0.5)
    assert len(_rank([a, b])) == 1


# --- before failing soft, an angle whose LLM call errored left no trace at all ---


def test_discover_records_the_failed_angle(monkeypatch):
    from beegent import run as runmod

    good = SearchAngle("good", "web_search", "r", dataset="d", format="GeoParquet")
    bad = SearchAngle("throttled", "web_search", "r", dataset="d", format="GeoPackage")

    def fake_resolve(angle):
        if angle.format == "GeoParquet":
            return make_candidate("https://ok.example/p", "https://ok.example/f.parquet"), None
        # what resolve_angle now returns when the agent aborts mid-loop
        return None, Candidate(
            url="https://slow.example/p", title="T", source="geofetch",
            claim={"failure_reason": "agent aborted: RateLimitError: 429"},
            cost={"steps_used": 3, "http_requests": 2, "prompt_tokens": 300,
                  "completion_tokens": 60, "total_tokens": 360})

    monkeypatch.setattr(runmod, "plan", lambda c, u, f=None: [good, bad])
    monkeypatch.setattr(runmod, "resolve_angle", fake_resolve)
    monkeypatch.setattr(runmod, "run_critic",
                        lambda *a: {"decision": "needs_human_review", "note": "n"})
    run = runmod.discover("Atlantis", "land cover")

    out = run.to_dict()
    assert len(out["candidates"]) == 1
    assert len(out["unresolved"]) == 1, "the throttled angle vanished"
    assert "429" in out["unresolved"][0]["claim"]["failure_reason"]
    # and its spend is still attributed in the run totals
    assert out["totals"]["total_tokens"] == 360
    assert out["totals"]["angles_run"] == 2


# --- run.totals summed only Candidate.cost, so planner and critic tokens were invisible ---

PLAN = {"angles": [{"description": "d", "url": "https://ok.example/p",
                    "dataset": "d", "format": "GeoParquet"}]}


def _install(monkeypatch, critic_decision="replan"):
    """A fake connector metering 1,000 planner tokens and 400 critic tokens per call."""
    plan, verdict = PLAN, {"decision": critic_decision, "note": "n"}

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

    monkeypatch.setattr(llm, "_connector", C())


def _verified():
    return Candidate(
        url="https://ok.example/p", title="T", source="geofetch", confidence=0.95,
        resource_url="https://ok.example/f.parquet",
        cost={"http_requests": 4, "prompt_tokens": 5_000,
              "completion_tokens": 1_000, "total_tokens": 6_000}), None


def test_planner_bucket_reaches_totals_on_the_early_ok_exit(monkeypatch, reset_llm_usage):
    """The gate passes, so discover() returns at its first exit - the common path."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    run = runmod.discover("Atlantis", "land cover")

    t = run.to_dict()["totals"]
    assert t["by_role"]["planner"]["total_tokens"] == 1_000
    assert t["by_role"]["geofetch"]["total_tokens"] == 6_000
    assert "critic" not in t["by_role"], "the gate passed, so no critic call"
    assert t["total_tokens"] == 7_000, "run-wide total = sum of the buckets"


def test_cached_tokens_reach_totals_from_both_metering_routes(monkeypatch, reset_llm_usage):
    """chat_json meters per role, chat_tools rides on Candidate.cost - both must carry it."""
    from beegent import run as runmod

    _install(monkeypatch)
    cached = Candidate(
        url="https://ok.example/p", title="T", source="geofetch", confidence=0.95,
        resource_url="https://ok.example/f.parquet",
        cost={"http_requests": 4, "prompt_tokens": 5_000, "completion_tokens": 1_000,
              "total_tokens": 6_000, "cached_tokens": 4_200})
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (cached, None))
    run = runmod.discover("Atlantis", "land cover")

    t = run.to_dict()["totals"]
    assert t["by_role"]["geofetch"]["cached_tokens"] == 4_200
    assert t["cached_tokens"] == 4_200, "the planner reported none of its own"
    assert t["total_tokens"] == 7_000, "cached must not inflate the run total"


def test_a_backend_reporting_no_cache_leaves_the_key_at_zero(monkeypatch, reset_llm_usage):
    """The key is always present, so a run can be read without knowing the backend."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    run = runmod.discover("Atlantis", "land cover")
    assert run.to_dict()["totals"]["cached_tokens"] == 0


def test_a_raising_critic_call_does_not_lose_the_run(monkeypatch, reset_llm_usage):
    """Every stage fails soft: a throttled critic must not discard work already paid for."""
    from beegent import run as runmod

    class Throttled(LLMConnector):
        provider = "fake"
        DEFAULT_MODELS = {"planner": "p", "geofetch": "g", "critic": "c"}
        def chat_json(self, model, messages):
            if model == "p":
                return PLAN, TokenUsage(800, 200)
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
    monkeypatch.setattr(llm, "_connector", Throttled())
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (None, dead))
    run = runmod.discover("Atlantis", "land cover")

    assert run.status == "needs_human_review"
    assert "RuntimeError" in run.reason, "the reason must say the critic was throttled"
    out = run.to_dict()
    assert out["totals"]["by_role"]["geofetch"]["total_tokens"] == 26_190, \
        "work already paid for must survive"
    assert len(out["unresolved"]) == 1, "the dead end must not vanish"


def test_critic_bucket_is_counted_when_the_gate_fires(monkeypatch, reset_llm_usage):
    from beegent import run as runmod

    _install(monkeypatch, critic_decision="needs_human_review")
    dead = Candidate(url="https://x.example/p", title="T", source="geofetch",
                     claim={"failure_reason": "nothing"},
                     cost={"prompt_tokens": 900, "completion_tokens": 100,
                           "total_tokens": 1_000})
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (None, dead))
    run = runmod.discover("Atlantis", "land cover")

    t = run.to_dict()["totals"]
    assert run.status == "needs_human_review"
    assert t["by_role"]["critic"]["total_tokens"] == 400
    assert t["total_tokens"] == 1_000 + 1_000 + 400


def test_geofetch_is_not_double_counted(monkeypatch, reset_llm_usage):
    """Geofetch rides on Candidate.cost and is kept out of the meter, so nothing doubles."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    run = runmod.discover("Atlantis", "land cover")

    assert "geofetch" not in llm.usage_by_role()
    assert run.totals["by_role"]["geofetch"]["total_tokens"] == 6_000


def test_meter_does_not_leak_between_runs(monkeypatch, reset_llm_usage):
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    first = runmod.discover("Atlantis", "land cover")
    second = runmod.discover("Atlantis", "land cover")
    assert first.totals["by_role"]["planner"] == second.totals["by_role"]["planner"]


# --- cross-run memory: nothing a run learns may vanish with it ---


def _store(tmp_path):
    from beegent.store import SqliteStore
    return SqliteStore(str(tmp_path / "m.db"))


def test_a_run_is_recorded_on_the_early_ok_exit(monkeypatch, tmp_path, reset_llm_usage):
    """The common path. Four exits route through _finalize(); each must persist."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    store = _store(tmp_path)
    run = runmod.discover("Atlantis", "land cover", store=store)

    assert run.status == "ok"
    assert len(store.prior_runs("Atlantis", 5)) == 1
    assert len(store.verified_links("Atlantis")) == 1


def test_the_critic_note_is_recorded_not_just_consumed(monkeypatch, tmp_path, reset_llm_usage):
    """A replan note used to be assigned to `feedback` and dropped - never stored anywhere."""
    from beegent import run as runmod

    _install(monkeypatch, critic_decision="needs_human_review")
    dead = Candidate(url="https://x.example/p", title="T", source="geofetch",
                     claim={"failure_reason": "nothing"}, cost={"total_tokens": 10})
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (None, dead))
    store = _store(tmp_path)
    run = runmod.discover("Atlantis", "land cover", store=store)

    assert run.critic_note == "n", "the verdict must reach the run object"
    assert store.prior_runs("Atlantis", 5)[0]["critic_note"] == "n"


def test_tried_accumulates_across_iterations(monkeypatch, tmp_path, reset_llm_usage):
    """run.unresolved is REPLACED each iteration, so collecting only at the end loses one."""
    from beegent import run as runmod

    _install(monkeypatch)  # replan -> two iterations
    seen = []

    def fake_resolve(angle):
        seen.append(1)
        return None, Candidate(url=f"https://x.example/{len(seen)}", title="T",
                               source="geofetch", claim={"failure_reason": "dead"},
                               cost={"total_tokens": 10})

    monkeypatch.setattr(runmod, "resolve_angle", fake_resolve)
    store = _store(tmp_path)
    runmod.discover("Atlantis", "land cover", store=store)

    tried = store.prior_runs("Atlantis", 5)[0]["tried"]
    assert len(tried) == 2, "iteration 1's dead end must survive iteration 2"


def test_prior_advice_reaches_the_planner_as_feedback(monkeypatch, tmp_path, reset_llm_usage):
    """The whole point: run N+1 starts knowing what run N found dead."""
    from beegent import run as runmod

    store = _store(tmp_path)
    store.record_run(
        DiscoveryRun(country="Atlantis", use_case="land cover", critic_note="try the bulk server"),
        [{"url": "https://dead.example/", "failure_reason": "unreachable"}])

    _install(monkeypatch)
    seen = {}
    monkeypatch.setattr(runmod, "plan", lambda c, u, f=None: seen.setdefault("feedback", f) and [])
    runmod.discover("Atlantis", "land cover", store=store)

    assert "try the bulk server" in seen["feedback"]
    assert "https://dead.example/" in seen["feedback"]


def test_a_broken_store_does_not_fail_the_run(monkeypatch, reset_llm_usage):
    """Memory is a nicety; every stage fails soft and this is no exception."""
    from beegent import run as runmod

    class Broken:
        def prior_runs(self, country, limit):
            return []
        def record_run(self, run, tried):
            raise RuntimeError("disk on fire")
        def verified_links(self, country):
            return []

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    run = runmod.discover("Atlantis", "land cover", store=Broken())
    assert run.status == "ok"


# --- a fresh catalog hit answers the run outright: no planner, no agent, no tokens ---


def _catalog_hit(days_old):
    from datetime import datetime, timedelta, timezone
    when = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return Candidate(url="https://page.example/p", title="boundaries", dataset="boundaries",
                     source="catalog", confidence=0.95,
                     resource_url="https://a.example/f.gpkg",
                     verification={"ok": True, "status": 206}, verified_at=when)


def test_a_fresh_catalog_hit_skips_planning_entirely(monkeypatch, reset_llm_usage):
    """The waste this exists to remove: rediscovering a link found three days ago."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "query_catalogs", lambda *a, **k: [_catalog_hit(3)])
    monkeypatch.setattr(runmod, "plan", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the planner must not run for a fresh catalog hit")))
    run = runmod.discover("Atlantis", "land cover")

    assert run.status == "ok"
    assert len(run.candidates) == 1
    assert run.totals["total_tokens"] == 0, "a cached answer costs no LLM call"


def test_a_stale_catalog_hit_still_discovers(monkeypatch, reset_llm_usage):
    """Live is not current: an old link is a floor, not an answer."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "query_catalogs", lambda *a, **k: [_catalog_hit(400)])
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    planned = []
    real_plan = runmod.plan
    monkeypatch.setattr(runmod, "plan", lambda *a, **k: planned.append(1) or real_plan(*a, **k))
    runmod.discover("Atlantis", "land cover")
    assert planned, "a stale hit must not short-circuit"


def test_the_window_can_be_switched_off(monkeypatch, reset_llm_usage):
    """CATALOG_FRESH_DAYS = 0 is the escape hatch."""
    from beegent import run as runmod

    monkeypatch.setattr(config, "CATALOG_FRESH_DAYS", 0)
    _install(monkeypatch)
    monkeypatch.setattr(runmod, "query_catalogs", lambda *a, **k: [_catalog_hit(0)])
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    planned = []
    real_plan = runmod.plan
    monkeypatch.setattr(runmod, "plan", lambda *a, **k: planned.append(1) or real_plan(*a, **k))
    runmod.discover("Atlantis", "land cover")
    assert planned


def test_a_hit_without_a_timestamp_never_short_circuits(monkeypatch, reset_llm_usage):
    from beegent import run as runmod

    hit = _catalog_hit(1)
    hit.verified_at = ""
    _install(monkeypatch)
    monkeypatch.setattr(runmod, "query_catalogs", lambda *a, **k: [hit])
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: _verified())
    planned = []
    real_plan = runmod.plan
    monkeypatch.setattr(runmod, "plan", lambda *a, **k: planned.append(1) or real_plan(*a, **k))
    runmod.discover("Atlantis", "land cover")
    assert planned


def test_a_cached_answer_is_not_recorded_as_a_new_discovery(monkeypatch, tmp_path,
                                                            reset_llm_usage):
    """Re-recording it would reset last_verified, and the freshness window would never expire."""
    from beegent import run as runmod

    _install(monkeypatch)
    monkeypatch.setattr(runmod, "query_catalogs", lambda *a, **k: [_catalog_hit(1)])
    store = _store(tmp_path)
    runmod.discover("Atlantis", "land cover", store=store)

    assert len(store.prior_runs("Atlantis", 5)) == 1, "the run itself is still recorded"
    assert store.verified_links("Atlantis") == [], "but the cache hit is not a new link"


def test_the_critic_sees_routes_earlier_RUNS_already_burned(monkeypatch, tmp_path,
                                                            reset_llm_usage):
    """It recommended the same publisher three Japan runs running, because it could not see them."""
    from beegent import run as runmod

    store = _store(tmp_path)
    store.record_run(DiscoveryRun(country="Atlantis", use_case="land cover"),
                     [{"url": "https://burned.example/last-time", "failure_reason": "dead"}])

    _install(monkeypatch, critic_decision="needs_human_review")
    dead = Candidate(url="https://fresh.example/this-time", title="T", source="geofetch",
                     claim={"failure_reason": "also dead"}, cost={"total_tokens": 10})
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (None, dead))
    seen = {}
    monkeypatch.setattr(runmod, "run_critic",
                        lambda *a: seen.setdefault("before", a[6]) and None
                        or {"decision": "needs_human_review", "note": "n"})
    runmod.discover("Atlantis", "land cover", store=store)

    assert seen["before"][0] == "https://fresh.example/this-time", "this run comes first"
    assert "https://burned.example/last-time" in seen["before"], "and the earlier run is there too"


def test_an_empty_store_passes_only_this_runs_urls(monkeypatch, reset_llm_usage):
    from beegent import run as runmod

    _install(monkeypatch, critic_decision="needs_human_review")
    dead = Candidate(url="https://only.example/now", title="T", source="geofetch",
                     claim={"failure_reason": "dead"}, cost={"total_tokens": 10})
    monkeypatch.setattr(runmod, "resolve_angle", lambda a: (None, dead))
    seen = {}
    monkeypatch.setattr(runmod, "run_critic",
                        lambda *a: seen.setdefault("before", a[6]) and None
                        or {"decision": "needs_human_review", "note": "n"})
    runmod.discover("Atlantis", "land cover")

    assert seen["before"] == ["https://only.example/now"]
