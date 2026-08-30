#!/usr/bin/env python3
"""config.py's environment overrides."""

import importlib
import os
import unittest
from unittest import mock

from beegent import config


class TestEnvOverrides(unittest.TestCase):
    """config.py reads os.environ in its module body, so every case reloads it."""

    def tearDown(self):
        # A leaked override would break test_geofetch/test_run, which read config.* live.
        importlib.reload(config)

    @staticmethod
    def _reload(**env):
        with mock.patch.dict(os.environ, env, clear=False):
            return importlib.reload(config)

    def test_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = importlib.reload(config)
        self.assertEqual(cfg.MAX_ANGLES, 3)

    def test_env_var_overrides(self):
        cfg = self._reload(MAX_ANGLES="1", TOOL_RESULT_MAX_CHARS="40000")
        self.assertEqual(cfg.MAX_ANGLES, 1)
        self.assertEqual(cfg.TOOL_RESULT_MAX_CHARS, 40000)

    def test_non_integer_fails_at_import(self):
        with self.assertRaises(SystemExit) as ctx:
            self._reload(MAX_ANGLES="tree")
        self.assertIn("MAX_ANGLES", str(ctx.exception))

    def test_below_one_fails_at_import(self):
        with self.assertRaises(SystemExit):
            self._reload(GEOFETCH_MAX_STEPS="0")

    def test_probe_bytes_is_not_overridable(self):
        """16 is a correctness floor - the geopackage magic signature is 16 bytes."""
        cfg = self._reload(PROBE_BYTES="8")
        self.assertEqual(cfg.PROBE_BYTES, 16)


if __name__ == "__main__":
    unittest.main()
