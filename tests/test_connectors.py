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


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(ValueError) as ctx:
        create_connector("gpt5-turbo-max")
    for name in ("ollama", "databricks", "groq", "mistral"):
        assert name in str(ctx.value)


@pytest.mark.parametrize("name", sorted(CONNECTORS))
def test_every_connector_satisfies_the_contract(name):
    cls = CONNECTORS[name]
    assert issubclass(cls, LLMConnector)
    assert cls.provider == name, "provider must match its registry key"
    for method in ("chat_json", "chat_tools", "validate"):
        assert callable(getattr(cls, method, None)), method


def test_the_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LLMConnector()


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


def test_override_is_used_by_chat_json():
    _, usage = _Odd().chat_json("m", [])
    assert (usage.prompt_tokens, usage.completion_tokens) == (70, 30)


def test_override_is_used_by_chat_tools_too():
    """Both call sites must route through the one seam, or an override half-works."""
    _, usage = _Odd().chat_tools("m", [], [])
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


@pytest.mark.parametrize("method", ("chat_json", "chat_tools"))
def test_cached_tokens_are_read_from_the_nested_field(mocker, method):
    c = _Cached()
    c.details = mocker.Mock(cached_tokens=768)
    args = ("m", []) if method == "chat_json" else ("m", [], [])
    _, usage = getattr(c, method)(*args)
    assert usage.cached_tokens == 768


def test_a_backend_that_does_not_report_them_reads_zero():
    _, usage = _Cached().chat_json("m", [])  # details stays None
    assert usage.cached_tokens == 0


def test_a_dict_shaped_usage_payload_is_read_too():
    """Only the serialization differs; a silent 0 here would be indistinguishable from none."""
    c = _Cached()
    c.details = {"cached_tokens": 512}
    _, usage = c.chat_json("m", [])
    assert usage.cached_tokens == 512


def test_cached_is_a_subset_of_prompt_not_an_addition(mocker):
    """Double-counting it would inflate every total the run reports."""
    c = _Cached()
    c.details = mocker.Mock(cached_tokens=900)
    _, usage = c.chat_json("m", [])
    assert usage.prompt_tokens == 1_000
    assert usage.total_tokens == 1_040, "cached is already inside prompt_tokens"


def test_add_accumulates_cached_across_calls():
    total = TokenUsage()
    total.add(TokenUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=60))
    total.add(TokenUsage(prompt_tokens=100, completion_tokens=10, cached_tokens=80))
    assert total.cached_tokens == 140
    assert total.total_tokens == 220


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


def test_groq_reports_a_missing_key_without_a_request(monkeypatch, mocker):
    monkeypatch.setenv("GROQ_API_KEY", "")
    mocker.patch("requests.get", side_effect=AssertionError("must not be called"))
    assert "GROQ_API_KEY" in GroqConnector(log=lambda m: None).validate()


def test_groq_reports_an_unreachable_endpoint_rather_than_raising(monkeypatch, mocker):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    mocker.patch("requests.get", side_effect=OSError("boom"))
    problem = GroqConnector(log=lambda m: None).validate()
    assert "OSError" in problem
    assert "GROQ_API_KEY" in problem


def test_groq_names_a_role_model_it_no_longer_serves(monkeypatch, mocker):
    """A retired id must fail here, not as a 404 mid-angle after tokens are spent."""
    served = mocker.Mock(**{"json.return_value": {"data": [{"id": "openai/gpt-oss-20b"}]}})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    mocker.patch("requests.get", return_value=served)
    problem = GroqConnector(log=lambda m: None).validate()
    assert GroqConnector.DEFAULT_MODELS["geofetch"] in problem
    # and what to switch to, or the message costs a round trip to act on
    assert "openai/gpt-oss-20b" in problem


def test_groq_validates_the_override_not_the_default(monkeypatch, mocker):
    """model_for() is the seam, so --geofetch-model is what actually gets checked."""
    every = [{"id": m} for m in set(GroqConnector.DEFAULT_MODELS.values())]
    served = mocker.Mock(**{"json.return_value": {"data": every}})
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    mocker.patch("requests.get", return_value=served)
    c = GroqConnector(log=lambda m: None)
    assert c.validate() is None
    monkeypatch.setenv("GEOFETCH_MODEL", "retired-model")
    assert "retired-model" in c.validate()


def test_mistral_reports_a_missing_key_without_a_request(monkeypatch, mocker):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    mocker.patch("requests.get", side_effect=AssertionError("must not be called"))
    assert "MISTRAL_API_KEY" in MistralConnector(log=lambda m: None).validate()


def test_mistral_reports_an_unreachable_endpoint_rather_than_raising(monkeypatch, mocker):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    mocker.patch("requests.get", side_effect=OSError("boom"))
    problem = MistralConnector(log=lambda m: None).validate()
    assert "OSError" in problem
    assert "MISTRAL_API_KEY" in problem


def test_mistral_names_a_role_model_the_endpoint_does_not_list(monkeypatch, mocker):
    """Catches a typo or a retired id - NOT entitlement, which /v1/models does not report."""
    served = mocker.Mock(**{"json.return_value": {"data": [{"id": "mistral-tiny-latest"}]}})
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    mocker.patch("requests.get", return_value=served)
    problem = MistralConnector(log=lambda m: None).validate()
    for model in set(MistralConnector.DEFAULT_MODELS.values()):
        assert model in problem  # derived, so retuning a default cannot break this
    assert "mistral-tiny-latest" in problem, "and what it could use instead"


def test_mistral_validates_the_override_not_the_default(monkeypatch, mocker):
    every = [{"id": m} for m in set(MistralConnector.DEFAULT_MODELS.values())]
    served = mocker.Mock(**{"json.return_value": {"data": every}})
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    mocker.patch("requests.get", return_value=served)
    c = MistralConnector(log=lambda m: None)
    assert c.validate() is None
    monkeypatch.setenv("GEOFETCH_MODEL", "not-a-model")
    assert "not-a-model" in c.validate()


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


def test_a_connector_missing_a_default_says_which_role():
    class Incomplete(OllamaConnector):
        DEFAULT_MODELS = {"planner": "x"}

    with pytest.raises(ValueError) as ctx:
        Incomplete(log=lambda m: None).model_for("critic")
    assert "critic" in str(ctx.value)
    assert "CRITIC_MODEL" in str(ctx.value)


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


def test_cli_beats_environment(banner):
    line = banner(["--geofetch-model", "from-cli"], {"GEOFETCH_MODEL": "from-env"})
    assert "geofetch=from-cli" in line


def test_environment_wins_when_no_flag_given(banner):
    assert "geofetch=from-env" in banner([], {"GEOFETCH_MODEL": "from-env"})


def test_connector_default_when_neither(banner):
    default = OllamaConnector(log=lambda m: None).DEFAULT_MODELS["geofetch"]
    assert f"geofetch={default}" in banner([])


def test_overriding_one_role_leaves_the_others(banner):
    line = banner(["--planner-model", "p-cli"])
    assert "planner=p-cli" in line
    assert "geofetch=" + OllamaConnector.DEFAULT_MODELS["geofetch"] in line


def test_invalid_backend_names_the_valid_ones(monkeypatch):
    from beegent import run as runmod

    monkeypatch.setattr(logging, "basicConfig", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run.py"] + BASE + ["--backend", "nope"])
    with pytest.raises(SystemExit):
        runmod.main()


def test_select_backend_replaces_the_lazy_default(monkeypatch):
    from beegent import llm

    monkeypatch.setattr(llm, "_connector", None)
    assert isinstance(llm.get_connector(), OllamaConnector)
    assert llm.get_connector() is llm.get_connector(), "built once, not per call"
    llm.select_backend("databricks")
    assert isinstance(llm.get_connector(), DatabricksConnector)


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


def test_retry_sums_both_attempts():
    c = _Retrying([("not json at all", 100, 20), ('{"a": 1}', 100, 25)])
    data, usage = c.chat_json("m", [])
    assert c.calls == 2, "the first reply was unparseable, so it retried"
    assert data == {"a": 1}
    assert usage.prompt_tokens == 200, "the discarded attempt still cost input"
    assert usage.completion_tokens == 45


def test_total_failure_still_reports_what_it_spent():
    c = _Retrying([("nope", 100, 20), ("still nope", 100, 20)])
    data, usage = c.chat_json("m", [])
    assert data is None, "None means no answer"
    assert usage.total_tokens == 240, "a run that answered nothing still billed"
