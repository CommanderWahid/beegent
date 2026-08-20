"""The web layer - one provider, imported directly by name, no pluggability layer."""

from .web_tools import WebTools, normalize_url

__all__ = ["WebTools", "normalize_url"]
