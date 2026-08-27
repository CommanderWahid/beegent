"""Databricks Model Serving / Unity AI Gateway - the worked example of adding a backend."""

import os
from typing import Callable

from beegent.connectors.base import OpenAICompatConnector


def serving_base_url(host: str) -> str:
    """Normalize a workspace host into the OpenAI-compatible serving base URL."""
    host = (host or "").removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{host}/serving-endpoints"


class DatabricksConnector(OpenAICompatConnector):
    provider = "databricks"
    # Claude serving endpoints 400 on response_format; parse_json() copes without it.
    supports_response_format = False

    # Serving-endpoint names, NOT bare model names - a 404 means it is not named this.
    DEFAULT_MODELS = {
        "planner": "databricks-claude-sonnet-4-6",
        "geofetch": "databricks-claude-haiku-4-5",  # the expensive loop: needs native
        # function calling, and runs GEOFETCH_MAX_STEPS times per angle
        "critic": "databricks-claude-opus-5",  # rare calls, highest stakes
    }

    def __init__(self, log: Callable[[str], None] = print) -> None:
        self.host = os.environ.get("DATABRICKS_HOST", "")
        self.token = os.environ.get("DATABRICKS_TOKEN", "")
        # TODO: PAT for now; move to OAuth machine-to-machine before production.
        super().__init__(base_url=serving_base_url(self.host), api_key=self.token, log=log)

    def _create(self, model: str, messages: list[dict], **kwargs):
        """Same call, with one endpoint quirk handled."""
        from openai import BadRequestError

        try:
            return super()._create(model, messages, **kwargs)
        except BadRequestError as exc:
            if "temperature" not in kwargs or "temperature" not in str(exc):
                raise
            self.log(f"  [llm] {model} rejects the temperature parameter; retrying without it")
            kwargs.pop("temperature")
            return super()._create(model, messages, **kwargs)

    # A metric this gateway spells differently gets a _token_usage() override here.

    def validate(self) -> str | None:
        if not self.host:
            return "databricks backend needs DATABRICKS_HOST (see .env.example)."
        if not self.token:
            return "databricks backend needs DATABRICKS_TOKEN (see .env.example)."
        return None
