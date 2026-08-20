"""
Catalog workers: deterministic (no LLM) lookups against known open data catalogs.
Deliberately empty for now.

Contract if this is ever built: whatever it returns lands in `run.candidates` alongside
geofetch's output, so it must uphold the same guarantee - a Candidate with a `resource_url`
must also carry a `verification` block from an actual probe. Use
`core.search_backends.WebTools.probe_url()` for that; do not hand back a catalog's advertised
download URL untested, since nothing downstream checks it.
"""

from core.schemas import Candidate


def run_catalog_workers(country: str, use_case: str) -> list[Candidate]:
    print("  [catalog] no catalog workers configured (catalog layer not yet designed)")
    return []
