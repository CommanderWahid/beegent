#!/usr/bin/env python3
"""The planner: how a model reply becomes SearchAngles, and what gets dropped."""

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


def run_plan(monkeypatch, angles, country="France",
             use_case="building footprints as a geoparquet file"):
    """plan()'s ANGLES with chat_json stubbed - the suite never calls a model."""
    reply = ({"angles": angles}, TokenUsage())
    monkeypatch.setattr(planner, "chat_json", lambda role, messages: reply)
    return plan(country, use_case)[0]


# --- angle validation: behaviour that already existed and was unpinned ---


def test_the_prompt_forbids_inventing_opaque_identifiers():
    """The only guard on this is the prompt, so pin that it is still in it."""
    assert "NEVER invent an OPAQUE IDENTIFIER" in planner.SYSTEM


def test_an_angle_without_a_url_is_dropped(monkeypatch):
    assert len(run_plan(monkeypatch, [angle(url=""), angle()])) == 1


def test_a_non_http_url_is_dropped(monkeypatch):
    assert run_plan(monkeypatch, [angle(url="ftp://x.example/data/")]) == []


def test_an_angle_without_a_description_is_dropped(monkeypatch):
    assert run_plan(monkeypatch, [{"url": DEEP}]) == []


def test_the_list_is_capped_at_max_angles(monkeypatch):
    got = run_plan(monkeypatch, [angle() for _ in range(config.MAX_ANGLES + 3)])
    assert len(got) == config.MAX_ANGLES


def test_a_missing_dataset_falls_back_to_the_description(monkeypatch):
    got = run_plan(monkeypatch, [angle(dataset="")])
    assert got[0].dataset == "national mapping agency bulk server"


def test_a_raising_call_yields_no_angles_rather_than_propagating(monkeypatch):
    """No angles is a survivable run that escalates; a traceback loses the whole run."""
    def boom(role, messages):
        raise RuntimeError("429")

    monkeypatch.setattr(planner, "chat_json", boom)
    assert plan("France", "buildings")[0] == []


def test_no_answer_yields_no_angles_rather_than_raising(monkeypatch):
    """chat_json returning None means 'no answer', never a negative answer."""
    monkeypatch.setattr(planner, "chat_json", lambda role, messages: (None, TokenUsage()))
    assert plan("France", "buildings")[0] == []


def test_the_prompt_treats_a_delivery_channel_as_not_a_format():
    """A use case naming an API/WFS names a CHANNEL; inventing a format filters out the data.

    Observed: a run that asked for "an OGC API-Features or WFS endpoint" had GeoJSON invented
    for it, which then rejected a verified 632 MB national boundary file for being GML.
    """
    assert "DELIVERY CHANNEL" in planner.SYSTEM
    assert "Naming a channel is not naming a format" in planner.SYSTEM
    # The narrow reading is the whole risk control: a named format must still win.
    assert "and no format" in planner.SYSTEM
    assert "If the use case named one, use exactly that" in planner.SYSTEM


def test_an_explicitly_empty_format_survives_planning(monkeypatch):
    """The invariant the channel rule depends on - "" must not be substituted downstream."""
    got = run_plan(monkeypatch, [angle(format="")])
    assert got[0].format == "", "an empty format disarms the magic-byte check; keep it empty"


def test_a_failed_call_reports_why_it_produced_nothing(monkeypatch):
    """A provider outage and an unplannable use case must not look alike downstream."""
    monkeypatch.setattr(planner, "chat_json",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503 capacity")))
    angles, reason = plan("France", "buildings")

    assert angles == []
    assert "RuntimeError" in reason and "503 capacity" in reason


def test_a_successful_call_reports_no_reason(monkeypatch):
    monkeypatch.setattr(planner, "chat_json", lambda *a, **k: ({"angles": [angle()]}, TokenUsage(1, 1)))
    assert plan("France", "buildings")[1] == ""
