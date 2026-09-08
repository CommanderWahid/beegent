#!/usr/bin/env python3
"""The connector layer: contract, registry, and the quirks each backend owns."""

import logging
import sys
from types import SimpleNamespace

import pytest

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
from tests.conftest import ANGLE, HAPPY_PATH, FakeLLM, build_tools

SHIPPED = (("ollama", OllamaConnector), ("databricks", DatabricksConnector),
           ("groq", GroqConnector), ("mistral", MistralConnector))


def _completion(content='{"a": 1}', **usage):
    """One chat completion, shaped like the SDK's - a namespace fabricates no attribute."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(**usage) if usage else None,
    )


# --- the registry ---


@pytest.mark.parametrize("name,cls", SHIPPED)
def test_every_shipped_connector_is_registered(name, cls):
    assert isinstance(create_connector(name), cls)


@pytest.mark.parametrize("name", sorted(CONNECTORS))
def test_every_connector_satisfies_the_contract(name):
    cls = CONNECTORS[name]
    assert issubclass(cls, LLMConnector)
    assert cls.provider == name, "provider must match its registry key"
    for method in ("chat_json", "chat_tools", "validate"):
        assert callable(getattr(cls, method, None)), method


def test_validate_is_part_of_the_contract_not_an_optional_hook():
    """A connector with no validate() of its own must fail at construction, not mid-run."""

    class NoValidate(LLMConnector):
        provider = "noval"
        def chat_json(self, model, messages):
            return {}, TokenUsage()
        def chat_tools(self, model, messages, tools):
            return None, TokenUsage()

    with pytest.raises(TypeError) as ctx:
        NoValidate()
    assert "validate" in str(ctx.value)


# --- usage extraction: field names are a provider convention, not a standard ---


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
        return _completion(input_tokens=70, output_tokens=30,
                           prompt_tokens=0, completion_tokens=0)


@pytest.mark.parametrize("method,args", [("chat_json", ("m", [])),
                                         ("chat_tools", ("m", [], []))])
def test_a_token_usage_override_is_used_by_both_call_paths(method, args):
    """Both must route through the one seam, or an override half-works."""
    _, usage = getattr(_Odd(), method)(*args)
    assert (usage.prompt_tokens, usage.completion_tokens) == (70, 30)


# --- cached prefixes are free against a TPM budget, so a run must be able to see them ---


class _Cached(OpenAICompatConnector):
    provider = "cached"
    DEFAULT_MODELS = {"planner": "m", "geofetch": "m", "critic": "m"}
    details = None

    def __init__(self):
        super().__init__("http://x", "k", log=lambda m: None)

    def validate(self):
        return None

    def _create(self, model, messages, **kw):
        return _completion(prompt_tokens=1_000, completion_tokens=40,
                           prompt_tokens_details=self.details)


@pytest.mark.parametrize("details,expected", [
    pytest.param("MOCK", 768, id="object"),          # OpenAI's nesting
    pytest.param({"cached_tokens": 512}, 512, id="dict"),  # only the serialization differs
    pytest.param(None, 0, id="not-reported"),        # and a 0 must not raise
])
def test_cached_tokens_are_read_from_whatever_shape_arrives(details, expected, mocker):
    c = _Cached()
    c.details = mocker.Mock(cached_tokens=768) if details == "MOCK" else details
    _, usage = c.chat_json("m", [])
    assert usage.cached_tokens == expected


def test_cached_is_a_subset_of_prompt_not_an_addition(mocker):
    """Double-counting it would inflate every total the run reports."""
    c = _Cached()
    c.details = mocker.Mock(cached_tokens=900)
    _, usage = c.chat_json("m", [])
    assert usage.prompt_tokens == 1_000
    assert usage.total_tokens == 1_040, "cached is already inside prompt_tokens"


# --- backend quirks: each used to be an `if config.LLM_BACKEND == ...` in shared code ---


def test_response_format_is_opted_into_per_backend():
    # Databricks' Claude endpoints 400 on response_format; Ollama and Groq accept it.
    assert OllamaConnector.supports_response_format
    assert GroqConnector.supports_response_format
    assert MistralConnector.supports_response_format
    assert not DatabricksConnector.supports_response_format


def test_databricks_retries_without_temperature(monkeypatch, mocker):
    """Opus rejects `temperature`; Sonnet and Haiku accept it. Ask, then drop."""
    from openai import BadRequestError

    c = DatabricksConnector(log=lambda m: None)
    calls = []

    def fake_create(model, messages, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise BadRequestError("temperature is not supported",
                                  response=mocker.Mock(status_code=400), body=None)
        return _completion()

    monkeypatch.setattr(OpenAICompatConnector, "_create", staticmethod(fake_create))
    c._create("databricks-claude-opus-5", [], temperature=0)
    assert len(calls) == 2, "should have retried once"
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1], "the retry must drop it"


def test_a_non_temperature_bad_request_is_not_swallowed(monkeypatch, mocker):
    from openai import BadRequestError

    c = DatabricksConnector(log=lambda m: None)

    def always_fails(model, messages, **kw):
        raise BadRequestError("model not found",
                              response=mocker.Mock(status_code=400), body=None)

    monkeypatch.setattr(OpenAICompatConnector, "_create", staticmethod(always_fails))
    with pytest.raises(BadRequestError):
        c._create("nope", [], temperature=0)


# --- validate(): fail before the run, not as a 404 mid-angle ---


def test_databricks_reports_missing_credentials(monkeypatch):
    """Credentials are read by the connector from os.environ, not from config."""
    monkeypatch.setenv("DATABRICKS_HOST", "")
    monkeypatch.setenv("DATABRICKS_TOKEN", "t")
    assert "DATABRICKS_HOST" in DatabricksConnector().validate()
    monkeypatch.setenv("DATABRICKS_HOST", "x.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "")
    assert "DATABRICKS_TOKEN" in DatabricksConnector().validate()


#: The two key-only hosted backends. Same four behaviours, so they are one test each.
KEY_ONLY = [
    pytest.param(GroqConnector, "GROQ_API_KEY", "gsk_x", "openai/gpt-oss-20b", id="groq"),
    pytest.param(MistralConnector, "MISTRAL_API_KEY", "k", "mistral-tiny-latest", id="mistral"),
]


@pytest.mark.parametrize("cls,env,key,other", KEY_ONLY)
@pytest.mark.parametrize("set_key", (False, True), ids=("no-key", "unreachable"))
def test_validate_reports_a_problem_rather_than_raising(cls, env, key, other, set_key,
                                                        monkeypatch, mocker):
    """No key must not even reach the wire; an unreachable endpoint must not propagate."""
    monkeypatch.setenv(env, key if set_key else "")
    mocker.patch("requests.get", side_effect=OSError("boom") if set_key
                 else AssertionError("no key: must not be called"))
    assert env in cls(log=lambda m: None).validate()


@pytest.mark.parametrize("cls,env,key,other", KEY_ONLY)
def test_a_role_model_the_endpoint_does_not_list_is_named(cls, env, key, other, monkeypatch,
                                                          mocker):
    """A retired id must fail here, not as a 404 mid-angle after tokens are spent."""
    served = mocker.Mock(**{"json.return_value": {"data": [{"id": other}]}})
    monkeypatch.setenv(env, key)
    mocker.patch("requests.get", return_value=served)
    problem = cls(log=lambda m: None).validate()
    for model in set(cls.DEFAULT_MODELS.values()):
        assert model in problem  # derived, so retuning a default cannot break this
    assert other in problem, "and what it could use instead"


@pytest.mark.parametrize("cls,env,key,other", KEY_ONLY)
def test_the_override_is_validated_not_the_default(cls, env, key, other, monkeypatch, mocker):
    """model_for() is the seam, so --geofetch-model is what actually gets checked."""
    every = [{"id": m} for m in set(cls.DEFAULT_MODELS.values())]
    served = mocker.Mock(**{"json.return_value": {"data": every}})
    monkeypatch.setenv(env, key)
    mocker.patch("requests.get", return_value=served)
    c = cls(log=lambda m: None)
    assert c.validate() is None
    monkeypatch.setenv("GEOFETCH_MODEL", "retired-model")
    assert "retired-model" in c.validate()


@pytest.mark.parametrize("given", ("x.cloud.databricks.com", "https://x.cloud.databricks.com",
                                   "https://x.cloud.databricks.com/"))
def test_host_accepts_bare_or_full_url(given):
    assert serving_base_url(given) == "https://x.cloud.databricks.com/serving-endpoints"


# --- role resolution: callers ask for a ROLE, the connector owns which model that means ---


@pytest.mark.parametrize("name,role", [(n, r) for n in sorted(CONNECTORS)
                                       for r in LLMConnector.ROLES])
def test_each_connector_resolves_every_role(name, role):
    c = CONNECTORS[name](log=lambda m: None)
    assert c.model_for(role), f"{name} has no model for {role}"


def test_env_var_overrides_the_default(monkeypatch):
    c = OllamaConnector(log=lambda m: None)
    assert c.model_for("planner") != "some-other-model"
    monkeypatch.setenv("PLANNER_MODEL", "some-other-model")
    assert c.model_for("planner") == "some-other-model"
    # an override for one role must not bleed into the others
    assert c.model_for("critic") == c.DEFAULT_MODELS["critic"]


def test_unknown_role_raises_rather_than_returning_none():
    """A typo must fail at the call, not become a request against an empty model."""
    c = OllamaConnector(log=lambda m: None)
    with pytest.raises(ValueError) as ctx:
        c.model_for("plannner")
    assert "planner" in str(ctx.value)  # lists the valid roles


# --- extensibility: the claim this layer exists to support, exercised rather than asserted ---


def test_a_third_party_connector_drives_the_pipeline(monkeypatch):
    from beegent.pipeline import geofetch as gf

    fake = FakeLLM(list(HAPPY_PATH))

    class MyConnector(LLMConnector):          # NOT OpenAI-shaped
        provider = "mine"
        def chat_json(self, model, messages):
            return {"decision": "replan", "note": "n"}, TokenUsage(11, 7)
        def chat_tools(self, model, messages, tools):
            return fake(model, messages, tools)
        def validate(self):
            return None                       # nothing to check, said explicitly

    monkeypatch.setitem(CONNECTORS, "mine", MyConnector)
    c = create_connector("mine")
    assert c.validate() is None               # it opted into "nothing to check"
    tools = build_tools()
    monkeypatch.setattr("beegent.pipeline.geofetch.WebTools", lambda **kw: tools)
    monkeypatch.setattr("beegent.pipeline.geofetch.chat_tools", c.chat_tools)
    found, missed = gf.resolve_angle(ANGLE, log=lambda m: None)
    assert found is not None, "a third-party connector should resolve an angle"
    assert found.verification["payload_type"] == "parquet"
    assert found.cost["total_tokens"] > 0


def test_a_new_backend_never_touches_config():
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
    assert [c.model_for(r) for r in c.ROLES] == ["p-1", "g-1", "c-1"]
    # and config.py names no backend beyond the default selector value
    src = inspect.getsource(cfg)
    # not "llama": the default LLM_BACKEND value "ollama" contains it
    for token in ("databricks", "deepseek", "qwen3", "claude", "groq", "gpt-oss",
                  "mistral", "magistral", "selfcontained"):
        assert token not in src, f"config.py should not mention {token!r}"


def test_usage_is_normalized_even_when_a_backend_reports_none(monkeypatch):
    c = OllamaConnector(log=lambda m: None)
    monkeypatch.setattr(OllamaConnector, "_create",
                        lambda *a, **k: _completion(content=None))
    _, usage = c.chat_tools("m", [], [])
    assert isinstance(usage, TokenUsage)
    assert usage.total_tokens == 0


# --- CLI precedence: CLI flag > environment > .env > the connector's DEFAULT_MODELS ---

BASE = ["--country", "X", "--use-case", "Y", "--out", "/dev/null"]


@pytest.fixture
def banner(monkeypatch, caplog):
    """The [config] line main() prints, for a given argv and environment."""
    def _banner(argv, env=None):
        from beegent import run as runmod
        from beegent.schemas import DiscoveryRun

        # main() configures ROOT logging, which would leak into every later test.
        monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
        # These cases test model RESOLUTION, not reachability: validate() pings a live
        # Ollama, which passes on a dev box and fails everywhere else.
        monkeypatch.setattr(OllamaConnector, "validate", lambda self: None)
        # Derived, so a new cost key cannot break this test by omission.
        stub = DiscoveryRun(country="X", use_case="Y", totals=dict.fromkeys(
            ("angles_run",) + runmod._COST_KEYS, 0) | {"by_role": {}})
        for key in ("GEOFETCH_MODEL", "PLANNER_MODEL", "CRITIC_MODEL"):
            monkeypatch.setenv(key, "")
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sys, "argv", ["run.py"] + BASE + argv)
        monkeypatch.setattr(runmod, "discover", lambda c, u: stub)
        with caplog.at_level(logging.INFO, logger="beegent.run"):
            runmod.main()
        return next(r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("[config]"))
    return _banner


DEFAULT_GEOFETCH = OllamaConnector.DEFAULT_MODELS["geofetch"]


@pytest.mark.parametrize("argv,env,expect", [
    pytest.param(["--geofetch-model", "cli"], {"GEOFETCH_MODEL": "env"}, "geofetch=cli",
                 id="cli-beats-env"),
    pytest.param([], {"GEOFETCH_MODEL": "env"}, "geofetch=env", id="env-beats-default"),
    pytest.param([], {}, f"geofetch={DEFAULT_GEOFETCH}", id="connector-default"),
    # an override for one role must not bleed into another
    pytest.param(["--planner-model", "p"], {}, f"geofetch={DEFAULT_GEOFETCH}", id="one-role-only"),
])
def test_the_precedence_chain_is_cli_then_env_then_default(argv, env, expect, banner):
    assert expect in banner(argv, env)


def test_invalid_backend_names_the_valid_ones(monkeypatch):
    from beegent import run as runmod

    monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run.py"] + BASE + ["--backend", "nope"])
    with pytest.raises(SystemExit):
        runmod.main()


# --- CHAT_JSON_ATTEMPTS makes a non-JSON reply cost twice ---


class _Retrying(OpenAICompatConnector):
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
        return _completion(content, prompt_tokens=prompt, completion_tokens=completion,
                           prompt_tokens_details=None)


@pytest.mark.parametrize("second,answer,total", [
    ('{"a": 1}', {"a": 1}, 245),   # recovered - but the discarded attempt still cost input
    ("still nope", None, 240),     # None means no answer, and it was still billed
])
def test_a_retry_sums_both_attempts(second, answer, total):
    """A discarded non-JSON reply cost real tokens; omitting it made a retry look free."""
    c = _Retrying([("not json at all", 100, 20), (second, 100, 25 if answer else 20)])
    data, usage = c.chat_json("m", [])
    assert c.calls == 2 and data == answer
    assert usage.total_tokens == total
