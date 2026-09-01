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


def test_probe_bytes_is_not_overridable(monkeypatch):
    """16 is a correctness floor - the geopackage magic signature is 16 bytes."""
    assert _reload(monkeypatch, PROBE_BYTES="8").PROBE_BYTES == 16
