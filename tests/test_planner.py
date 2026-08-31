#!/usr/bin/env python3
"""The planner: how a model reply becomes SearchAngles, and what gets dropped."""

import unittest
from unittest import mock

from beegent import config
from beegent.pipeline import planner
from beegent.pipeline.planner import plan
from beegent.schemas import TokenUsage

DEEP = "https://cadastre.example/data/etalab-cadastre/latest/"


def angle(**over):
    base = {"description": "national mapping agency bulk server", "channel_hint": "catalog",
            "rationale": "bulk file server", "url": DEEP,
            "dataset": "building footprints, whole country",
            "format": "GeoParquet", "vintage": "latest"}
    return {**base, **over}


def run_plan(angles, country="France", use_case="building footprints as a geoparquet file"):
    """plan() with chat_json stubbed - the suite never calls a model."""
    reply = ({"angles": angles}, TokenUsage())
    with mock.patch.object(planner, "chat_json", lambda role, messages: reply):
        return plan(country, use_case)


class TestAngleValidation(unittest.TestCase):
    """Behaviour that already existed and was unpinned - the planner had no test file."""

    def test_the_prompt_forbids_inventing_opaque_identifiers(self):
        """The only guard on this is the prompt, so pin that it is still in it."""
        self.assertIn("NEVER invent an OPAQUE IDENTIFIER", planner.SYSTEM)

    def test_an_angle_without_a_url_is_dropped(self):
        got = run_plan([angle(url=""), angle()])
        self.assertEqual(len(got), 1)

    def test_a_non_http_url_is_dropped(self):
        self.assertEqual(run_plan([angle(url="ftp://x.example/data/")]), [])

    def test_an_angle_without_a_description_is_dropped(self):
        self.assertEqual(run_plan([{"url": DEEP}]), [])

    def test_the_list_is_capped_at_max_angles(self):
        self.assertEqual(len(run_plan([angle() for _ in range(config.MAX_ANGLES + 3)])),
                         config.MAX_ANGLES)

    def test_a_missing_dataset_falls_back_to_the_description(self):
        got = run_plan([angle(dataset="")])
        self.assertEqual(got[0].dataset, "national mapping agency bulk server")

    def test_a_raising_call_yields_no_angles_rather_than_propagating(self):
        """No angles is a survivable run that escalates; a traceback loses the whole run."""
        with mock.patch.object(planner, "chat_json", side_effect=RuntimeError("429")):
            self.assertEqual(plan("France", "buildings"), [])

    def test_no_answer_yields_no_angles_rather_than_raising(self):
        """chat_json returning None means 'no answer', never a negative answer."""
        with mock.patch.object(planner, "chat_json", lambda role, messages: (None, TokenUsage())):
            self.assertEqual(plan("France", "buildings"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
