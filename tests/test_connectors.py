#!/usr/bin/env python3
"""The connector layer: contract, registry, and the quirks each backend owns."""

import os
import unittest
from unittest import mock

from beegent.connectors import (
    CONNECTORS,
    DatabricksConnector,
    GroqConnector,
    LLMConnector,
    MistralConnector,
    OllamaConnector,
    OpenAICompatConnector,
    create_connector,
)
from beegent.connectors.databricks import serving_base_url
from beegent.schemas import TokenUsage
from tests.fixtures import ANGLE, HAPPY_PATH, FakeLLM, make_tools


class TestRegistry(unittest.TestCase):
    def test_every_shipped_connector_is_registered(self):
        for name, cls in (("ollama", OllamaConnector), ("databricks", DatabricksConnector),
                          ("groq", GroqConnector), ("mistral", MistralConnector)):
            with self.subTest(provider=name):
                self.assertIsInstance(create_connector(name), cls)

    def test_unknown_backend_names_the_alternatives(self):
        with self.assertRaises(ValueError) as ctx:
            create_connector("gpt5-turbo-max")
        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("databricks", str(ctx.exception))
        self.assertIn("groq", str(ctx.exception))
        self.assertIn("mistral", str(ctx.exception))

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

    def test_validate_is_part_of_the_contract_not_an_optional_hook(self):
        """A connector with no validate() of its own must fail at construction, not mid-run."""

        class NoValidate(LLMConnector):
            provider = "noval"
            def chat_json(self, model, messages):
                return {}, TokenUsage()
            def chat_tools(self, model, messages, tools):
                return None, TokenUsage()

        with self.assertRaises(TypeError) as ctx:
            NoValidate()
        self.assertIn("validate", str(ctx.exception))


class TestTokenUsageExtractionIsAConnectorConcern(unittest.TestCase):
    """Field names for usage are a provider convention, not a standard."""

    class _Odd(OpenAICompatConnector):
        provider = "odd"
        DEFAULT_MODELS = {"planner": "m", "geofetch": "m", "critic": "m"}

        def __init__(self):
            super().__init__("http://x", "k", log=lambda m: None)

        def validate(self):
            return None

        def _token_usage(self, raw):
            # this backend says input_tokens/output_tokens
            return TokenUsage(prompt_tokens=getattr(raw, "input_tokens", 0) or 0,
                              completion_tokens=getattr(raw, "output_tokens", 0) or 0)

        def _create(self, model, messages, **kw):
            return mock.Mock(
                choices=[mock.Mock(message=mock.Mock(content='{"a": 1}'))],
                usage=mock.Mock(input_tokens=70, output_tokens=30,
                                prompt_tokens=0, completion_tokens=0),
            )

    def test_override_is_used_by_chat_json(self):
        _, usage = self._Odd().chat_json("m", [])
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (70, 30))

    def test_override_is_used_by_chat_tools_too(self):
        """Both call sites must route through the one seam, or an override half-works."""
        _, usage = self._Odd().chat_tools("m", [], [])
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (70, 30))


class TestCachedTokens(unittest.TestCase):
    """Cached prompt prefixes are free against a TPM budget, so a run must be able to see them."""

    class _Cached(OpenAICompatConnector):
        provider = "cached"
        DEFAULT_MODELS = {"planner": "m", "geofetch": "m", "critic": "m"}
        details = None

        def __init__(self):
            super().__init__("http://x", "k", log=lambda m: None)

        def validate(self):
            return None

        def _create(self, model, messages, **kw):
            return mock.Mock(
                choices=[mock.Mock(message=mock.Mock(content='{"a": 1}'))],
                usage=mock.Mock(prompt_tokens=1_000, completion_tokens=40,
                                prompt_tokens_details=self.details),
            )

    def test_cached_tokens_are_read_from_the_nested_field(self):
        c = self._Cached()
        c.details = mock.Mock(cached_tokens=768)
        for call in (lambda: c.chat_json("m", []), lambda: c.chat_tools("m", [], [])):
            with self.subTest(call=call):
                _, usage = call()
                self.assertEqual(usage.cached_tokens, 768)

    def test_a_backend_that_does_not_report_them_reads_zero(self):
        _, usage = self._Cached().chat_json("m", [])  # details stays None
        self.assertEqual(usage.cached_tokens, 0)

    def test_a_dict_shaped_usage_payload_is_read_too(self):
        """Only the serialization differs; a silent 0 here would be indistinguishable from none."""
        c = self._Cached()
        c.details = {"cached_tokens": 512}
        _, usage = c.chat_json("m", [])
        self.assertEqual(usage.cached_tokens, 512)

    def test_cached_is_a_subset_of_prompt_not_an_addition(self):
        """Double-counting it would inflate every total the run reports."""
        c = self._Cached()
        c.details = mock.Mock(cached_tokens=900)
        _, usage = c.chat_json("m", [])
        self.assertEqual(usage.prompt_tokens, 1_000)
        self.assertEqual(usage.total_tokens, 1_040, "cached is already inside prompt_tokens")

    def test_add_accumulates_cached_across_calls(self):
        total = TokenUsage()
        total.add(TokenUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=60))
        total.add(TokenUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=80))
        self.assertEqual(total.cached_tokens, 140)
        self.assertEqual(total.total_tokens, 220)


class TestBackendQuirksAreOwnedLocally(unittest.TestCase):
    """Each quirk used to be an `if config.LLM_BACKEND == ...` in shared code."""

    def test_response_format_is_opted_into_per_backend(self):
        # Databricks' Claude endpoints 400 on response_format; Ollama and Groq accept it.
        self.assertTrue(OllamaConnector.supports_response_format)
        self.assertTrue(GroqConnector.supports_response_format)
        self.assertTrue(MistralConnector.supports_response_format)
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

    def test_groq_reports_a_missing_key_without_a_request(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}), \
             mock.patch("requests.get", side_effect=AssertionError("must not be called")):
            self.assertIn("GROQ_API_KEY", GroqConnector(log=lambda m: None).validate())

    def test_groq_reports_an_unreachable_endpoint_rather_than_raising(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_x"}), \
             mock.patch("requests.get", side_effect=OSError("boom")):
            problem = GroqConnector(log=lambda m: None).validate()
        self.assertIn("OSError", problem)
        self.assertIn("GROQ_API_KEY", problem)

    def test_groq_names_a_role_model_it_no_longer_serves(self):
        """A retired id must fail here, not as a 404 mid-angle after tokens are spent."""
        served = mock.Mock(**{"json.return_value": {"data": [{"id": "openai/gpt-oss-20b"}]}})
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_x"}), \
             mock.patch("requests.get", return_value=served):
            problem = GroqConnector(log=lambda m: None).validate()
        self.assertIn(GroqConnector.DEFAULT_MODELS["geofetch"], problem)
        # and what to switch to, or the message costs a round trip to act on
        self.assertIn("openai/gpt-oss-20b", problem)

    def test_groq_validates_the_override_not_the_default(self):
        """model_for() is the seam, so --geofetch-model is what actually gets checked."""
        every = [{"id": m} for m in set(GroqConnector.DEFAULT_MODELS.values())]
        served = mock.Mock(**{"json.return_value": {"data": every}})
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_x"}), \
             mock.patch("requests.get", return_value=served):
            c = GroqConnector(log=lambda m: None)
            self.assertIsNone(c.validate())
            with mock.patch.dict(os.environ, {"GEOFETCH_MODEL": "retired-model"}):
                self.assertIn("retired-model", c.validate())

    def test_mistral_reports_a_missing_key_without_a_request(self):
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": ""}), \
             mock.patch("requests.get", side_effect=AssertionError("must not be called")):
            self.assertIn("MISTRAL_API_KEY", MistralConnector(log=lambda m: None).validate())

    def test_mistral_reports_an_unreachable_endpoint_rather_than_raising(self):
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), \
             mock.patch("requests.get", side_effect=OSError("boom")):
            problem = MistralConnector(log=lambda m: None).validate()
        self.assertIn("OSError", problem)
        self.assertIn("MISTRAL_API_KEY", problem)

    def test_mistral_names_a_role_model_the_endpoint_does_not_list(self):
        """Catches a typo or a retired id - NOT entitlement, which /v1/models does not report."""
        served = mock.Mock(**{"json.return_value": {"data": [{"id": "mistral-small-latest"}]}})
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), \
             mock.patch("requests.get", return_value=served):
            problem = MistralConnector(log=lambda m: None).validate()
        for model in set(MistralConnector.DEFAULT_MODELS.values()):
            self.assertIn(model, problem)  # derived, so retuning a default cannot break this
        self.assertIn("mistral-small-latest", problem, "and what it could use instead")

    def test_mistral_validates_the_override_not_the_default(self):
        every = [{"id": m} for m in set(MistralConnector.DEFAULT_MODELS.values())]
        served = mock.Mock(**{"json.return_value": {"data": every}})
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), \
             mock.patch("requests.get", return_value=served):
            c = MistralConnector(log=lambda m: None)
            self.assertIsNone(c.validate())
            with mock.patch.dict(os.environ, {"GEOFETCH_MODEL": "not-a-model"}):
                self.assertIn("not-a-model", c.validate())

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
                return {"decision": "replan", "note": "n"}, TokenUsage(11, 7)
            def chat_tools(self, model, messages, tools):
                return llm(model, messages, tools)
            def validate(self):
                return None                       # nothing to check, said explicitly

        CONNECTORS["mine"] = MyConnector
        try:
            c = create_connector("mine")
            self.assertIsNone(c.validate())       # it opted into "nothing to check"
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
        """A connector owns its models and credentials; config.py never learns its name."""
        import inspect

        from beegent import config as cfg

        class SelfContained(OpenAICompatConnector):
            provider = "selfcontained"
            DEFAULT_MODELS = {"planner": "p-1", "geofetch": "g-1", "critic": "c-1"}
            def __init__(self, log=print):
                super().__init__(base_url="https://x/v1", api_key="k", log=log)
            def validate(self):
                return None

        c = SelfContained(log=lambda m: None)
        self.assertEqual([c.model_for(r) for r in c.ROLES], ["p-1", "g-1", "c-1"])
        # and config.py names no backend beyond the default selector value
        src = inspect.getsource(cfg)
        # not "llama": the default LLM_BACKEND value "ollama" contains it
        for token in ("databricks", "deepseek", "qwen3", "claude", "groq", "gpt-oss",
                      "mistral", "magistral", "selfcontained"):
            self.assertNotIn(token, src, f"config.py should not mention {token!r}")

    def test_usage_is_normalized_even_when_a_backend_reports_none(self):
        c = OllamaConnector(log=lambda m: None)
        with mock.patch.object(OllamaConnector, "_create",
                               lambda *a, **k: mock.Mock(usage=None, choices=[mock.Mock()])):
            _, usage = c.chat_tools("m", [], [])
        self.assertIsInstance(usage, TokenUsage)
        self.assertEqual(usage.total_tokens, 0)


class TestCliOverrides(unittest.TestCase):
    """Precedence: CLI flag > environment > .env > the connector's DEFAULT_MODELS."""

    BASE = ["--country", "X", "--use-case", "Y", "--out", "/dev/null"]

    def setUp(self):
        """main() configures ROOT logging, which would leak into every later test."""
        patcher = mock.patch("logging.basicConfig")  # not enterContext: that is 3.11+
        patcher.start()
        self.addCleanup(patcher.stop)

    def _banner(self, argv, env=None):
        import sys

        from beegent import run as runmod
        from beegent.schemas import DiscoveryRun

        # Derived, so a new cost key cannot break this test by omission.
        stub = DiscoveryRun(country="X", use_case="Y", totals=dict.fromkeys(
            ("angles_run",) + runmod._COST_KEYS, 0) | {"by_role": {}})
        clean = {k: "" for k in ("GEOFETCH_MODEL", "PLANNER_MODEL", "CRITIC_MODEL")}
        with mock.patch.dict(os.environ, {**clean, **(env or {})}), \
             mock.patch.object(sys, "argv", ["run.py"] + self.BASE + argv), \
             mock.patch.object(runmod, "discover", lambda c, u: stub):
            with self.assertLogs("beegent.run", level="INFO") as cm:
                runmod.main()
        return next(r.getMessage() for r in cm.records
                    if r.getMessage().startswith("[config]"))

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


class TestChatJsonChargesEveryAttempt(unittest.TestCase):
    """CHAT_JSON_ATTEMPTS makes a non-JSON reply cost twice."""

    class _Conn(OpenAICompatConnector):
        provider = "t"
        DEFAULT_MODELS = {"planner": "m", "geofetch": "m", "critic": "m"}

        def __init__(self, replies):
            super().__init__("http://x", "k", log=lambda m: None)
            self.replies = list(replies)
            self.calls = 0

        def validate(self):
            return None

        def _create(self, model, messages, **kw):
            content, prompt, completion = self.replies[self.calls]
            self.calls += 1
            return mock.Mock(
                choices=[mock.Mock(message=mock.Mock(content=content))],
                # None, not a bare Mock: a Mock fabricates every attribute asked of it.
                usage=mock.Mock(prompt_tokens=prompt, completion_tokens=completion,
                                prompt_tokens_details=None),
            )

    def test_retry_sums_both_attempts(self):
        c = self._Conn([("not json at all", 100, 20), ('{"a": 1}', 100, 25)])
        data, usage = c.chat_json("m", [])
        self.assertEqual(c.calls, 2, "the first reply was unparseable, so it retried")
        self.assertEqual(data, {"a": 1})
        self.assertEqual(usage.prompt_tokens, 200, "the discarded attempt still cost input")
        self.assertEqual(usage.completion_tokens, 45)

    def test_total_failure_still_reports_what_it_spent(self):
        c = self._Conn([("nope", 100, 20), ("still nope", 100, 20)])
        data, usage = c.chat_json("m", [])
        self.assertIsNone(data, "None means no answer")
        self.assertEqual(usage.total_tokens, 240, "a run that answered nothing still billed")
