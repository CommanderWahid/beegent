"""
Catalog workers: deterministic (no LLM) lookups against known open data catalogs.
Deliberately empty for now.
"""

from schemas import Candidate


def run_catalog_workers(country: str, use_case: str) -> list[Candidate]:
    print("  [catalog] no catalog workers configured (catalog layer not yet designed)")
    return []
