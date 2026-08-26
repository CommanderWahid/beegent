#!/usr/bin/env python3
"""The one LLM client: <think> stripping, which both Ollama defaults need."""

import unittest

from beegent.llm import strip_think

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


# --------------------------------------------------------------------------- #
# Agent-loop tests
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    unittest.main(verbosity=2)
