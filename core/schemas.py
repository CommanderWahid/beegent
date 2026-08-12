"""Data shapes for a discovery run. Plain dataclasses - no framework needed."""

import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse


@dataclass
class SearchAngle:
    description: str  # e.g. "national statistics office boundary data"
    channel_hint: str  # "catalog" | "web_search" | "national_geoportal"
    rationale: str


@dataclass
class Candidate:
    url: str
    title: str
    source: str  # "catalog:hdx" | "search_explore" etc.
    hops: int  # page-fetches it took to find this (0 = direct search/catalog hit)
    rationale: str
    confidence: float | None = None  # filled in by triage
    publisher: str | None = None  # rough guess at the publishing org, filled in by triage

    def publisher_key(self) -> str:
        """Normalized publisher, for the gate's diversity check.

        Publisher beats domain here: one national portal can host datasets from
        many independent agencies (diverse), while the same dataset mirrored
        across three domains is not (not diverse).

        Falls back to the domain when triage gave us no publisher, so unknowns
        don't all collapse into one bucket and fake a single-publisher result.

        TODO: this is exact-match-after-normalizing, so "data.gouv.fr" and
        "data.gouv.fr open data platform" count as two. Over-counting diversity
        is the safe direction - it errs towards not escalating a good result.
        """
        if self.publisher:
            key = re.sub(r"[^a-z0-9]+", " ", self.publisher.lower()).strip()
            if key:
                return key
        return urlparse(self.url).netloc.lower().removeprefix("www.")


@dataclass
class DiscoveryRun:
    country: str
    use_case: str
    iteration: int = 1  # bumps each time the critic sends it back to the planner
    max_iterations: int = 2  # hard cap - after this, stop re-planning and flag for human
    candidates: list[Candidate] = field(default_factory=list)
    # Found, but triage never produced a verdict on them (null confidence is the marker).
    # Excluded from candidates so junk can't ride in on a failed call, recorded here so
    # nothing that was found disappears silently.
    unresolved: list[Candidate] = field(default_factory=list)
    status: str = "ok"  # "ok" | "needs_human_review"
    reason: str | None = None  # populated when status is needs_human_review

    def to_dict(self) -> dict:
        return asdict(self)
