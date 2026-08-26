"""Databricks Model Serving / Unity AI Gateway - the worked example of adding a backend.

Read this file to see what implementing a connector actually costs: a base_url built from a
workspace host, a token, and the two quirks this backend has. Everything else is inherited.

Model names here are SERVING ENDPOINT names, not bare model names - a 404 usually means the
endpoint is not called that in your workspace.
"""

import os
from typing import Callable

from beegent.connectors.base import OpenAICompatConnector


def serving_base_url(host: str) -> str:
    """Normalize a workspace host into the OpenAI-compatible serving base URL.

    DATABRICKS_HOST may be pasted either as a bare hostname or as the full https:// URL
    shown in the UI; accept both rather than silently building an unresolvable base_url.
    """
    host = (host or "").removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{host}/serving-endpoints"


class DatabricksConnector(OpenAICompatConnector):
    provider = "databricks"
    # Claude serving endpoints 400 on response_format (INVALID_PARAMETER_VALUE). Every
    # prompt already asks for JSON-only output and parse_json() falls back to extracting
    # the outermost {...} from a fenced or prose reply, so leaving it off costs nothing.
    supports_response_format = False

    # Serving-endpoint names, NOT bare model names - verify with
    # `GET /api/2.0/serving-endpoints` if any of these ever 404 in your workspace.
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
        """Same call, with one endpoint quirk handled.

        Databricks' Claude endpoints differ from each other: Opus rejects `temperature`
        while Sonnet and Haiku accept it. Asking and then dropping the parameter beats
        keeping a per-model capability table in sync with a workspace's endpoints - and a
        hard failure here would take out the critic, whose whole job is rescuing a weak run.
        """
        from openai import BadRequestError

        try:
            return super()._create(model, messages, **kwargs)
        except BadRequestError as exc:
            if "temperature" not in kwargs or "temperature" not in str(exc):
                raise
            self.log(f"  [llm] {model} rejects the temperature parameter; retrying without it")
            kwargs.pop("temperature")
            return super()._create(model, messages, **kwargs)

    def validate(self) -> str | None:
        if not self.host:
            return "databricks backend needs DATABRICKS_HOST (see .env.example)."
        if not self.token:
            return "databricks backend needs DATABRICKS_TOKEN (see .env.example)."
        return None
