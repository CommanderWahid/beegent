"""Deterministic catalog lookups - the non-agentic route to candidates."""

import logging

from beegent.schemas import Candidate

_log = logging.getLogger(__name__)


def query_catalogs(country: str, use_case: str) -> list[Candidate]:
    _log.info("  [catalog] no catalogs configured (catalog layer not yet designed)")
    return []
