"""Deterministic catalog lookups - the non-agentic route to candidates."""

from beegent.schemas import Candidate


def query_catalogs(country: str, use_case: str) -> list[Candidate]:
    print("  [catalog] no catalogs configured (catalog layer not yet designed)")
    return []
