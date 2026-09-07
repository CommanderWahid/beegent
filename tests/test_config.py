#!/usr/bin/env python3
"""config.py's environment overrides."""

import importlib

import pytest

from beegent import config


@pytest.fixture(autouse=True)
def _restore_config():
    """A leaked override would break test_geofetch/test_run, which read config.* live."""
    yield
    importlib.reload(config)


def _reload(monkeypatch, **env):
    """config.py reads os.environ in its module body, so every case reloads it."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config)


def test_default_when_unset(clean_env):
    assert importlib.reload(config).MAX_ANGLES == 3


def test_env_var_overrides(monkeypatch):
    cfg = _reload(monkeypatch, MAX_ANGLES="1", TOOL_RESULT_MAX_CHARS="40000")
    assert cfg.MAX_ANGLES == 1
    assert cfg.TOOL_RESULT_MAX_CHARS == 40000


def test_non_integer_fails_at_import(monkeypatch):
    with pytest.raises(SystemExit) as ctx:
        _reload(monkeypatch, MAX_ANGLES="tree")
    assert "MAX_ANGLES" in str(ctx.value)


def test_below_one_fails_at_import(monkeypatch):
    with pytest.raises(SystemExit):
        _reload(monkeypatch, GEOFETCH_MAX_STEPS="0")


def test_the_relevance_thresholds_are_settable(monkeypatch):
    """A run recommends a value on a model change; applying it must not need a source edit."""
    cfg = _reload(monkeypatch, CATALOG_MIN_RELEVANCE="0.71", MEMORY_MIN_RELEVANCE="0.8")
    assert (cfg.CATALOG_MIN_RELEVANCE, cfg.MEMORY_MIN_RELEVANCE) == (0.71, 0.8)


def test_a_non_numeric_threshold_fails_at_import(monkeypatch):
    with pytest.raises(SystemExit) as ctx:
        _reload(monkeypatch, CATALOG_MIN_RELEVANCE="high")
    assert "CATALOG_MIN_RELEVANCE" in str(ctx.value)


def test_a_threshold_above_one_fails_at_import(monkeypatch):
    """No cosine reaches 1.5, so this would match nothing at all - in silence."""
    with pytest.raises(SystemExit):
        _reload(monkeypatch, MEMORY_MIN_RELEVANCE="1.5")


def test_zero_is_a_valid_threshold(monkeypatch):
    """0 is the documented off switch: filter nothing, offer everything stored."""
    assert _reload(monkeypatch, CATALOG_MIN_RELEVANCE="0").CATALOG_MIN_RELEVANCE == 0.0


def test_the_freshness_off_switch_works_from_the_environment(monkeypatch):
    """It was documented as an off switch while _int's floor of 1 rejected it."""
    assert _reload(monkeypatch, CATALOG_FRESH_DAYS="0").CATALOG_FRESH_DAYS == 0


def test_the_floor_of_one_still_holds_everywhere_else(monkeypatch):
    with pytest.raises(SystemExit):
        _reload(monkeypatch, MAX_ANGLES="0")


def test_probe_bytes_is_not_overridable(monkeypatch):
    """16 is a correctness floor - the geopackage magic signature is 16 bytes."""
    assert _reload(monkeypatch, PROBE_BYTES="8").PROBE_BYTES == 16


def test_the_store_path_is_expanded_at_import(monkeypatch):
    """A "~" set in .env gets no shell expansion, and would become a literal directory."""
    cfg = _reload(monkeypatch, BEEGENT_DB="~/somewhere/memory.db")
    assert not cfg.BEEGENT_DB.startswith("~")
    assert cfg.BEEGENT_DB.endswith("/somewhere/memory.db")


def test_the_store_has_a_default_path(clean_env):
    """Unset means on, not off - the feature is no longer dark unless you know a variable."""
    assert importlib.reload(config).BEEGENT_DB.endswith(".beegent/memory.db")
