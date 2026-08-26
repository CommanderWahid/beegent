"""
Deterministic (no LLM) lookups against known open data catalogs - the non-agentic route
to candidates, as opposed to geofetch reasoning its way to one. Deliberately empty for now.

Contract if this is ever built: whatever it returns lands in `run.candidates` alongside
geofetch's output, so it must uphold the same guarantee - a Candidate with a `resource_url`
must also carry a `verification` block from an actual probe. Use
`beegent.web_tools.WebTools.probe_url()` for that; do not hand back a catalog's advertised
download URL untested, since nothing downstream checks it.
"""

from beegent.schemas import Candidate


def query_catalogs(country: str, use_case: str) -> list[Candidate]:
    print("  [catalog] no catalogs configured (catalog layer not yet designed)")
    return []
