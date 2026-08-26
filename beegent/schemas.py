"""Data shapes for a discovery run. Plain dataclasses - no framework needed."""

from dataclasses import asdict, dataclass, field


@dataclass
class TokenUsage:
    """Normalized token accounting, summed across every LLM call in a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "TokenUsage | None") -> None:
        if other:
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens


@dataclass
class SearchAngle:
    description: str  # e.g. "national statistics office boundary data"
    channel_hint: str  # "catalog" | "web_search" | "national_geoportal"
    rationale: str
    # These four ARE the fetcher's parameters - one per GeofetchAgent.run() argument. The
    # planner's whole job is to map (country, use_case) into them, so nothing downstream has
    # to rediscover any of it. An angle missing `url` is not a fetch task and is dropped.
    url: str = ""  # where to start: a dataset page, bulk file server, or API base
    dataset: str = ""  # free-text description of the file wanted
    format: str = ""  # "GeoParquet" | "GeoPackage" | ... | "" = any, which disarms the
    # magic-byte guardrail down to a plain liveness check
    vintage: str = "latest"  # "latest", or a date/year/version


@dataclass
class Candidate:
    url: str
    title: str
    source: str  # "geofetch" | "catalog:hdx" etc.
    confidence: float | None = None  # geofetch's claim.confidence band, mapped through
    # config.CONFIDENCE_BY_REPORT. None marks an unresolved entry - see DiscoveryRun below.
    # The endpoint that actually serves the data - a direct download or an API. Distinct from
    # `url`, which stays whatever page discovery found and cited. Never guessed: geofetch only
    # sets this after independently probing it, so a non-None value here means bytes of the
    # right type were seen on the wire. None means no fetchable endpoint could be verified.
    resource_url: str | None = None
    # --- the trust split. These two blocks must never be conflated. ---
    # What the MODEL said: edition, vintage_date, file_size_bytes, checksum, its own
    # confidence band, the ordered `evidence` chain, and `failure_reason` on a dead end.
    # Every key here is unverified - useful for review and provenance, never a fact.
    claim: dict = field(default_factory=dict)
    # What the HARNESS measured by re-probing `resource_url`: status, payload_type,
    # first_bytes_hex, total_size_bytes. Written by deterministic code, never by a model -
    # this is the field to trust. None means nothing was verified.
    verification: dict | None = None
    # What the angle cost: steps_used, http_requests, and token counts. Measured, not claimed.
    cost: dict = field(default_factory=dict)


@dataclass
class DiscoveryRun:
    country: str
    use_case: str
    iteration: int = 1  # bumps each time the critic sends it back to the planner
    max_iterations: int = 2  # hard cap - after this, stop re-planning and flag for human
    backend: str = ""  # config.LLM_BACKEND - which wire the models were reached over
    models: dict = field(default_factory=dict)  # {planner, geofetch, critic} - one file
    # tells you what actually ran, without cross-referencing config against the environment
    totals: dict = field(default_factory=dict)  # run-wide cost, accumulated as angles finish
    candidates: list[Candidate] = field(default_factory=list)
    # Angles that dead-ended: reached a page, but no fetchable endpoint could be verified
    # (null confidence and null resource_url are the markers). Excluded from candidates so
    # nothing unverified rides in, recorded here so nothing found disappears silently.
    unresolved: list[Candidate] = field(default_factory=list)
    status: str = "ok"  # "ok" | "needs_human_review"
    reason: str | None = None  # populated when status is needs_human_review

    def to_dict(self) -> dict:
        return asdict(self)
