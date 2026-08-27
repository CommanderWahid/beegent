#!/usr/bin/env python3
"""The one LLM client: <think> stripping, which both Ollama defaults need."""

import unittest
from unittest import mock

from beegent import llm
from beegent.llm import strip_think
from beegent.schemas import TokenUsage

from tests.fixtures import HAPPY_PATH, PORTAL, run_agent, tool_turn


class TestStripThink(unittest.TestCase):
    def test_strip_think_removes_reasoning_blocks(self):
        raw = "<think>long hidden\nreasoning</think>Following the atom link."
        self.assertEqual(strip_think(raw), "Following the atom link.")
        self.assertEqual(strip_think(None), "")
        self.assertEqual(strip_think("no blocks here"), "no blocks here")

    def test_think_blocks_stripped_from_kept_history(self):
        turns = [tool_turn("fetch_page", {"url": PORTAL},
                           thought="<think>secret</think>recon")] + list(HAPPY_PATH)[1:]
        res, llm = run_agent(turns)
        self.assertTrue(res.found)
        assistants = [m for m in llm.seen_messages if m.get("role") == "assistant"]
        self.assertTrue(any(m["content"] == "recon" for m in assistants))
        self.assertFalse(any("<think>" in m.get("content", "") for m in assistants))


# --- agent-loop tests ---


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUsageMeter(unittest.TestCase):
    """chat_json's usage was dropped, so planner and critic tokens never reached totals."""

    def setUp(self):
        llm.reset_usage()
        self.addCleanup(llm.reset_usage)

    def _conn(self, usage):
        class C:
            ROLES = ("planner", "geofetch", "critic")
            def model_for(self, role):
                return f"model-{role}"
            def chat_json(self, model, messages):
                return {"ok": True}, usage
        return C()

    def test_chat_json_returns_the_same_2_tuple_shape_as_chat_tools(self):
        """Both are one completion reporting its cost, so they read alike everywhere."""
        with mock.patch.object(llm, "get_connector", lambda: self._conn(TokenUsage(5, 3))):
            data, usage = llm.chat_json("planner", [])
        self.assertEqual(data, {"ok": True})
        self.assertEqual(usage.total_tokens, 8)

    def test_every_call_site_unpacks_before_testing_the_answer(self):
        """The tuple is always truthy, so only an unpacked test is safe."""
        import inspect

        from beegent.pipeline import critic, planner

        for mod in (planner, critic):
            src = inspect.getsource(mod)
            self.assertIn("= chat_json(", src, mod.__name__)
            self.assertNotIn("if chat_json(", src, f"{mod.__name__} must unpack first")
            self.assertNotIn(") or {}", src, f"{mod.__name__} must not or-default the tuple")

    def test_usage_is_recorded_against_the_role(self):
        with mock.patch.object(llm, "get_connector", lambda: self._conn(TokenUsage(5, 3))):
            llm.chat_json("planner", [])
            llm.chat_json("critic", [])
            llm.chat_json("planner", [])
        by_role = llm.usage_by_role()
        self.assertEqual(by_role["planner"].total_tokens, 16)   # two calls
        self.assertEqual(by_role["critic"].total_tokens, 8)
        self.assertNotIn("geofetch", by_role, "chat_tools must not feed the meter")

    def test_reset_clears_between_runs(self):
        with mock.patch.object(llm, "get_connector", lambda: self._conn(TokenUsage(5, 3))):
            llm.chat_json("planner", [])
        llm.reset_usage()
        self.assertEqual(llm.usage_by_role(), {})
