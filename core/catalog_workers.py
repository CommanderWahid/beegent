"""Catalog workers: deterministic (no LLM) lookups against known open data catalogs.

Deliberately empty for now.

The previous version hardcoded two specific catalogs and a 12-entry country -> ISO3 table,
which meant every country nobody had thought to type in was silently skipped. That is the
opposite of what this tool is for, so the whole layer is parked rather than patched: which
catalogs to query, how a country maps onto each one's identifiers, and how that list is kept
fresh are design questions, not lookup tables to extend.

`run_catalog_workers()` keeps its signature so core/run.py is unaffected - the pipeline runs
on search & explore alone until this is designed.
"""

from schemas import Candidate


def run_catalog_workers(country: str, use_case: str) -> list[Candidate]:
    print("  [catalog] no catalog workers configured (catalog layer not yet designed)")
    return []
