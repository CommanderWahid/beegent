"""The connector registry."""

from beegent.connectors.base import LLMConnector, OpenAICompatConnector
from beegent.connectors.databricks import DatabricksConnector
from beegent.connectors.ollama import OllamaConnector

#: provider key -> class. `config.LLM_BACKEND` selects one of these.
CONNECTORS: dict[str, type[LLMConnector]] = {
    "ollama": OllamaConnector,  # the default: local, no key, no cost
    "databricks": DatabricksConnector,
}

__all__ = ["CONNECTORS", "DatabricksConnector", "LLMConnector", "OllamaConnector",
           "OpenAICompatConnector", "create_connector"]


def create_connector(provider: str, **kwargs) -> LLMConnector:
    """Build the connector for `provider`, or fail naming what is available."""
    try:
        cls = CONNECTORS[provider]
    except KeyError:
        raise ValueError(
            f"unknown LLM backend {provider!r}; choose from {sorted(CONNECTORS)}"
        ) from None
    return cls(**kwargs)
