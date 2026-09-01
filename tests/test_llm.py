#!/usr/bin/env python3
"""The one LLM client: <think> stripping, which both Ollama defaults need."""

from beegent import llm
from beegent.llm import strip_think
from beegent.schemas import TokenUsage

from tests.conftest import HAPPY_PATH, PORTAL, tool_turn


def test_strip_think_removes_reasoning_blocks():
    raw = "<think>long hidden\nreasoning</think>Following the atom link."
    assert strip_think(raw) == "Following the atom link."
    assert strip_think(None) == ""
    assert strip_think("no blocks here") == "no blocks here"


def test_think_blocks_stripped_from_kept_history(run_agent):
    turns = [tool_turn("fetch_page", {"url": PORTAL},
                       thought="<think>secret</think>recon")] + list(HAPPY_PATH)[1:]
    res, fake = run_agent(turns)
    assert res.found
    assistants = [m for m in fake.seen_messages if m.get("role") == "assistant"]
    assert any(m["content"] == "recon" for m in assistants)
    assert not any("<think>" in m.get("content", "") for m in assistants)


# --- the usage meter: chat_json's usage was dropped, so planner and critic
# --- tokens never reached totals ---


def _conn(usage):
    class C:
        ROLES = ("planner", "geofetch", "critic")
        def model_for(self, role):
            return f"model-{role}"
        def chat_json(self, model, messages):
            return {"ok": True}, usage
    return C()


def test_chat_json_returns_the_same_2_tuple_shape_as_chat_tools(monkeypatch, reset_llm_usage):
    """Both are one completion reporting its cost, so they read alike everywhere."""
    monkeypatch.setattr(llm, "get_connector", lambda: _conn(TokenUsage(5, 3)))
    data, usage = llm.chat_json("planner", [])
    assert data == {"ok": True}
    assert usage.total_tokens == 8


def test_every_call_site_unpacks_before_testing_the_answer():
    """The tuple is always truthy, so only an unpacked test is safe."""
    import inspect

    from beegent.pipeline import critic, planner

    for mod in (planner, critic):
        src = inspect.getsource(mod)
        assert "= chat_json(" in src, mod.__name__
        assert "if chat_json(" not in src, f"{mod.__name__} must unpack first"
        assert ") or {}" not in src, f"{mod.__name__} must not or-default the tuple"


def test_usage_is_recorded_against_the_role(monkeypatch, reset_llm_usage):
    monkeypatch.setattr(llm, "get_connector", lambda: _conn(TokenUsage(5, 3)))
    llm.chat_json("planner", [])
    llm.chat_json("critic", [])
    llm.chat_json("planner", [])
    by_role = llm.usage_by_role()
    assert by_role["planner"].total_tokens == 16   # two calls
    assert by_role["critic"].total_tokens == 8
    assert "geofetch" not in by_role, "chat_tools must not feed the meter"


def test_reset_clears_between_runs(monkeypatch, reset_llm_usage):
    monkeypatch.setattr(llm, "get_connector", lambda: _conn(TokenUsage(5, 3)))
    llm.chat_json("planner", [])
    llm.reset_usage()
    assert llm.usage_by_role() == {}
