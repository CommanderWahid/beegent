#!/usr/bin/env python3
"""The connector layer: contract, registry, and the quirks each backend owns.

The point of these tests is that adding a backend must be a subclass plus a registry entry.
test_a_third_party_connector_drives_the_pipeline is the real check - if registering a new
connector and running an angle through it is awkward, the abstraction is wrong.
"""

import os
import unittest
from unittest import mock

from beegent.connectors import (
    CONNECTORS,
    DatabricksConnector,
    LLMConnector,
    OllamaConnector,
    OpenAICompatConnector,
    create_connector,
)
from beegent.connectors.databricks import serving_base_url
from beegent.schemas import TokenUsage
from tests.fixtures import ANGLE, HAPPY_PATH, FakeLLM, make_tools


class TestRegistry(unittest.TestCase):
    def test_both_shipped_connectors_are_registered(self):
        self.assertIsInstance(create_connector("ollama"), OllamaConnector)
        self.assertIsInstance(create_connector("databricks"), DatabricksConnector)

    def test_unknown_backend_names_the_alternatives(self):
        with self.assertRaises(ValueError) as ctx:
            create_connector("gpt5-turbo-max")
        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("databricks", str(ctx.exception))

    def test_every_connector_satisfies_the_contract(self):
        for name, cls in CONNECTORS.items():
            with self.subTest(provider=name):
                self.assertTrue(issubclass(cls, LLMConnector))
                self.assertEqual(cls.provider, name, "provider must match its registry key")
                for method in ("chat_json", "chat_tools", "validate"):
                    self.assertTrue(callable(getattr(cls, method, None)), method)

    def test_the_abstract_base_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            LLMConnector()


class TestBackendQuirksAreOwnedLocally(unittest.TestCase):
    """Each quirk used to be an `if config.LLM_BACKEND == ...` in shared code."""

    def test_only_ollama_opts_into_json_mode(self):
        # Databricks' Claude endpoints 400 on response_format; Ollama accepts it.
        self.assertTrue(OllamaConnector.supports_response_format)
        self.assertFalse(DatabricksConnector.supports_response_format)

    def test_databricks_retries_without_temperature(self):
        """Opus rejects `temperature`; Sonnet and Haiku accept it. Ask, then drop."""
        from openai import BadRequestError

        c = DatabricksConnector(log=lambda m: None)
        calls = []

        def fake_create(model, messages, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise BadRequestError("temperature is not supported",
                                      response=mock.Mock(status_code=400), body=None)
            return mock.Mock()

        with mock.patch.object(OpenAICompatConnector, "_create", staticmethod(fake_create)):
            c._create("databricks-claude-opus-5", [], temperature=0)
        self.assertEqual(len(calls), 2, "should have retried once")
        self.assertIn("temperature", calls[0])
        self.assertNotIn("temperature", calls[1], "the retry must drop it")

    def test_a_non_temperature_bad_request_is_not_swallowed(self):
        from openai import BadRequestError

        c = DatabricksConnector(log=lambda m: None)

        def always_fails(model, messages, **kw):
            raise BadRequestError("model not found",
                                  response=mock.Mock(status_code=400), body=None)

        with mock.patch.object(OpenAICompatConnector, "_create", staticmethod(always_fails)):
            with self.assertRaises(BadRequestError):
                c._create("nope", [], temperature=0)


class TestValidate(unittest.TestCase):
    def test_databricks_reports_missing_credentials(self):
        """Credentials are read by the connector from os.environ, not from config."""
        with mock.patch.dict(os.environ, {"DATABRICKS_HOST": "", "DATABRICKS_TOKEN": "t"}):
            self.assertIn("DATABRICKS_HOST", DatabricksConnector().validate())
        with mock.patch.dict(os.environ, {"DATABRICKS_HOST": "x.cloud.databricks.com",
                                          "DATABRICKS_TOKEN": ""}):
            self.assertIn("DATABRICKS_TOKEN", DatabricksConnector().validate())

    def test_host_accepts_bare_or_full_url(self):
        want = "https://x.cloud.databricks.com/serving-endpoints"
        for given in ("x.cloud.databricks.com", "https://x.cloud.databricks.com",
                      "https://x.cloud.databricks.com/"):
            with self.subTest(host=given):
                self.assertEqual(serving_base_url(given), want)


class TestRoleResolution(unittest.TestCase):
    """Callers ask for a pipeline ROLE; the connector owns which model that means."""

    def test_each_connector_resolves_every_role(self):
        for name, cls in CONNECTORS.items():
            c = cls(log=lambda m: None)
            for role in c.ROLES:
                with self.subTest(provider=name, role=role):
                    self.assertTrue(c.model_for(role), f"{name} has no model for {role}")

    def test_env_var_overrides_the_default(self):
        c = OllamaConnector(log=lambda m: None)
        self.assertNotEqual(c.model_for("planner"), "some-other-model")
        with mock.patch.dict(os.environ, {"PLANNER_MODEL": "some-other-model"}):
            self.assertEqual(c.model_for("planner"), "some-other-model")
            # an override for one role must not bleed into the others
            self.assertEqual(c.model_for("critic"), c.DEFAULT_MODELS["critic"])

    def test_unknown_role_raises_rather_than_returning_none(self):
        """A typo must fail at the call, not become a request against an empty model."""
        c = OllamaConnector(log=lambda m: None)
        with self.assertRaises(ValueError) as ctx:
            c.model_for("plannner")
        self.assertIn("planner", str(ctx.exception))  # lists the valid roles

    def test_a_connector_missing_a_default_says_which_role(self):
        class Incomplete(OllamaConnector):
            DEFAULT_MODELS = {"planner": "x"}

        with self.assertRaises(ValueError) as ctx:
            Incomplete(log=lambda m: None).model_for("critic")
        self.assertIn("critic", str(ctx.exception))
        self.assertIn("CRITIC_MODEL", str(ctx.exception))


class TestExtensibility(unittest.TestCase):
    """The claim this layer exists to support, exercised rather than asserted."""

    def test_a_third_party_connector_drives_the_pipeline(self):
        from beegent.pipeline import geofetch as gf

        llm = FakeLLM(list(HAPPY_PATH))

        class MyConnector(LLMConnector):          # NOT OpenAI-shaped
            provider = "mine"
            def chat_json(self, model, messages):
                return {"decision": "replan", "note": "n"}
            def chat_tools(self, model, messages, tools):
                return llm(model, messages, tools)

        CONNECTORS["mine"] = MyConnector
        try:
            c = create_connector("mine")
            self.assertIsNone(c.validate())       # default is permissive
            tools = make_tools()
            with mock.patch("beegent.pipeline.geofetch.WebTools", lambda **kw: tools), \
                 mock.patch("beegent.pipeline.geofetch.chat_tools", c.chat_tools):
                found, missed = gf.resolve_angle(ANGLE, log=lambda m: None)
            self.assertIsNotNone(found, "a third-party connector should resolve an angle")
            self.assertEqual(found.verification["payload_type"], "parquet")
            self.assertGreater(found.cost["total_tokens"], 0)
        finally:
            CONNECTORS.pop("mine", None)

    def test_a_new_backend_never_touches_config(self):
        """The property this whole layout exists for: a connector declares its own models
        and credentials, so config.py never learns a backend's name."""
        import inspect

        from beegent import config as cfg

        class SelfContained(OpenAICompatConnector):
            provider = "selfcontained"
            DEFAULT_MODELS = {"planner": "p-1", "geofetch": "g-1", "critic": "c-1"}
            def __init__(self, log=print):
                super().__init__(base_url="https://x/v1", api_key="k", log=log)

        c = SelfContained(log=lambda m: None)
        self.assertEqual([c.model_for(r) for r in c.ROLES], ["p-1", "g-1", "c-1"])
        # and config.py names no backend beyond the default selector value
        src = inspect.getsource(cfg)
        for token in ("databricks", "deepseek", "qwen3", "claude", "selfcontained"):
            self.assertNotIn(token, src, f"config.py should not mention {token!r}")

    def test_usage_is_normalized_even_when_a_backend_reports_none(self):
        c = OllamaConnector(log=lambda m: None)
        with mock.patch.object(OllamaConnector, "_create",
                               lambda *a, **k: mock.Mock(usage=None, choices=[mock.Mock()])):
            _, usage = c.chat_tools("m", [], [])
        self.assertIsInstance(usage, TokenUsage)
        self.assertEqual(usage.total_tokens, 0)


class TestCliOverrides(unittest.TestCase):
    """Precedence: CLI flag > environment > .env > the connector's DEFAULT_MODELS.

    Driven through main()'s real argument handling rather than model_for() alone - the
    wiring is what is new here, not the resolution.
    """

    BASE = ["--country", "X", "--use-case", "Y", "--out", "/dev/null"]

    def _banner(self, argv, env=None):
        import contextlib
        import io
        import sys

        from beegent import run as runmod
        from beegent.schemas import DiscoveryRun

        stub = DiscoveryRun(country="X", use_case="Y", totals=dict.fromkeys(
            ("angles_run", "http_requests", "prompt_tokens", "completion_tokens",
             "total_tokens"), 0))
        clean = {k: "" for k in ("GEOFETCH_MODEL", "PLANNER_MODEL", "CRITIC_MODEL")}
        with mock.patch.dict(os.environ, {**clean, **(env or {})}), \
             mock.patch.object(sys, "argv", ["run.py"] + self.BASE + argv), \
             mock.patch.object(runmod, "discover", lambda c, u: stub):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                runmod.main()
        return next(l for l in buf.getvalue().splitlines() if l.startswith("[config]"))

    def test_cli_beats_environment(self):
        line = self._banner(["--geofetch-model", "from-cli"], {"GEOFETCH_MODEL": "from-env"})
        self.assertIn("geofetch=from-cli", line)

    def test_environment_wins_when_no_flag_given(self):
        self.assertIn("geofetch=from-env", self._banner([], {"GEOFETCH_MODEL": "from-env"}))

    def test_connector_default_when_neither(self):
        default = OllamaConnector(log=lambda m: None).DEFAULT_MODELS["geofetch"]
        self.assertIn(f"geofetch={default}", self._banner([]))

    def test_overriding_one_role_leaves_the_others(self):
        line = self._banner(["--planner-model", "p-cli"])
        self.assertIn("planner=p-cli", line)
        self.assertIn("geofetch=" + OllamaConnector.DEFAULT_MODELS["geofetch"], line)

    def test_invalid_backend_names_the_valid_ones(self):
        import sys
        from beegent import run as runmod

        with mock.patch.object(sys, "argv", ["run.py"] + self.BASE + ["--backend", "nope"]), \
             self.assertRaises(SystemExit):
            runmod.main()


class TestBackendSelection(unittest.TestCase):
    def test_select_backend_replaces_the_lazy_default(self):
        from beegent import llm

        original = llm._connector
        try:
            llm._connector = None
            self.assertIsInstance(llm.get_connector(), OllamaConnector)
            self.assertIs(llm.get_connector(), llm.get_connector(), "built once, not per call")
            llm.select_backend("databricks")
            self.assertIsInstance(llm.get_connector(), DatabricksConnector)
        finally:
            llm._connector = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
