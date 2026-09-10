#!/usr/bin/env python3
"""The web API: a pure consumer of beegent, gating what a run costs."""

import json
import threading

import pytest

# The web API is an optional extra, exactly like embeddings: absent is not an error.
# CI installs it (`uv sync --extra api`), so these never skip where it matters.
pytest.importorskip("fastapi", reason="install the 'api' extra to exercise the web API")
pytest.importorskip("pycountry", reason="install the 'api' extra to exercise the web API")

from api import chat, runner


@pytest.mark.parametrize("given,expected", [
    ("France", "France"), ("netherlands", "Netherlands"), ("NL", "Netherlands"),
    ("Taiwan", "Taiwan, Province of China"),   # in ISO 3166, so it is searchable
    # A typo is not a region: it resolves, and the confirmation card shows it before spending.
    ("Netherland", "Netherlands"), ("the Netherlands", "Netherlands"),
    ("Holland", "Netherlands"), ("Franc", "France"),
])
def test_a_real_country_resolves_to_its_canonical_name(given, expected):
    assert chat.real_country(given) == expected


@pytest.mark.parametrize("given",
                         ["Europe", "the Alps", "Catalonia", "Atlantis", "", "  ", "xyz"])
def test_a_region_or_a_non_country_is_rejected(given):
    """Guessing the geography wrong wastes the whole run - it is the one gap worth blocking on."""
    assert chat.real_country(given) == ""


def _reply(mocker, **fields):
    from beegent.schemas import TokenUsage
    mocker.patch.object(chat, "chat_json", return_value=(fields, TokenUsage(1, 1)))
    return chat.interpret("anything")


def test_a_search_missing_its_country_is_not_ready(mocker):
    out = _reply(mocker, intent="search", use_case="land cover", country="Europe")
    assert out["missing"] == ["country"] and not out["ready"]


def test_a_gated_search_says_what_it_needs_rather_than_promising_work(mocker):
    """The model writes "Got it" believing it had a country; only the harness knows it did not."""
    out = _reply(mocker, intent="search", use_case="land cover", country="Europe",
                 reply="Got it - I'll look for land cover.")
    assert "Got it" in out["reply"] and "'Europe'" in out["reply"]
    assert "which country" in out["reply"].lower()


def test_a_ready_search_reply_is_left_exactly_as_the_model_wrote_it(mocker):
    out = _reply(mocker, intent="search", use_case="land cover", country="France",
                 reply="Got it - searching France.")
    assert out["reply"] == "Got it - searching France."


def test_a_complete_search_is_ready_with_every_field(mocker):
    out = _reply(mocker, intent="search", use_case="land cover", country="france",
                 format="GeoPackage", vintage="2025")
    assert out["ready"] and out["missing"] == []
    assert (out["country"], out["format"]) == ("France", "GeoPackage")


def test_format_and_vintage_never_block_a_run(mocker):
    """Both have a defined 'unspecified' behaviour already; only geography is unrecoverable."""
    out = _reply(mocker, intent="search", use_case="land cover", country="France")
    assert out["ready"] and (out["format"], out["vintage"]) == ("", "")


def test_a_catalog_question_is_never_gated(mocker):
    out = _reply(mocker, intent="catalog", country="", use_case="")
    assert out["missing"] == [] and out["intent"] == "catalog"


def test_an_unreadable_reply_does_not_raise(mocker):
    """The front door fails soft like every other stage."""
    mocker.patch.object(chat, "chat_json", side_effect=RuntimeError("429"))
    assert chat.interpret("hi")["intent"] == "other"


@pytest.mark.parametrize("fields,expected", [
    # A channel is folded in as a CHANNEL - naming a format here would pin the guardrail.
    ({"use_case": "admin boundaries", "channel": "api"},
     "admin boundaries " + chat.API_PHRASE),
    ({"use_case": "building footprints", "format": "GeoPackage"},
     "building footprints as a GeoPackage file"),
    ({"use_case": "land cover", "format": "GeoTIFF", "vintage": "2024"},
     "land cover as a GeoTIFF file, 2024 edition"),
    # "latest" is the default the pipeline already assumes, so it adds nothing.
    ({"use_case": "land cover", "vintage": "latest"}, "land cover"),
    # Never say it twice when the user already did.
    ({"use_case": "footprints as a GeoPackage", "format": "GeoPackage"},
     "footprints as a GeoPackage"),
    ({"use_case": "boundaries from a WFS service", "channel": "api"},
     "boundaries from a WFS service"),
    # A named format wins: the user asked for a file, so the channel clause stays out.
    ({"use_case": "parcels", "channel": "api", "format": "GeoJSON"},
     "parcels as a GeoJSON file"),
])
def test_the_four_fields_are_folded_into_the_one_string_discover_reads(fields, expected):
    assert chat.compose_use_case(fields) == expected


def test_a_channel_never_becomes_a_format(mocker):
    """format pins the magic-byte guardrail; "an API" must not land there."""
    out = _reply(mocker, intent="search", use_case="admin boundaries", country="Kenya",
                 channel="api", format="")
    assert out["channel"] == "api" and out["format"] == ""
    assert out["sent_as"].endswith(chat.API_PHRASE)


def test_a_reader_that_arrives_late_still_gets_the_whole_log():
    """A queue drains: after a reload the Logs tab would show only what arrived next."""
    lines = runner._Lines()
    lines.add("[plan] 2 angle(s)")
    lines.add("  probe_url https://example.test/x.gpkg")
    lines.finish()
    replay = ["[plan] 2 angle(s)", "  probe_url https://example.test/x.gpkg"]
    assert list(lines.tail()) == replay and list(lines.tail()) == replay


def test_a_reader_blocks_for_the_rest_of_a_running_log():
    """Retaining lines must not cost the live tail - the log IS the progress indicator."""
    lines = runner._Lines()
    lines.add("[plan] 2 angle(s)")
    threading.Timer(0.05, lambda: (lines.add("[gate] ok"), lines.finish())).start()
    assert list(lines.tail()) == ["[plan] 2 angle(s)", "[gate] ok"]


def test_the_resolved_config_names_models_and_never_a_key(mocker):
    """The CLI prints this line; the UI needs the same answer, minus anything secret."""
    from api import main
    out = main.resolved_config()
    assert out["backend"] and set(out["models"]) == {"planner", "geofetch", "critic"}
    assert "key" not in json.dumps(out).lower() and "token" not in json.dumps(out).lower()


def test_stopping_an_unknown_run_is_a_404_not_a_silent_success():
    """The button must never report a stop it did not perform."""
    from fastapi import HTTPException

    from api import main
    with pytest.raises(HTTPException) as caught:
        main.stop_run("no-such-run")
    assert caught.value.status_code == 404


def test_stop_sets_the_flag_discover_reads():
    r = runner.Runner()
    r._stops["abc"] = threading.Event()
    assert r.stop("abc") and r._stops["abc"].is_set()
    assert not r.stop("nope")
