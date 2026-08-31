"""Data shapes for a discovery run. Plain dataclasses - no framework needed."""

from dataclasses import asdict, dataclass, field


@dataclass
class TokenUsage:
    """Normalized token accounting for one or more LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # a SUBSET of prompt_tokens - a prefix the backend served from cache

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens  # cached is inside prompt, never added

    def add(self, other: "TokenUsage | None") -> None:
        if other:
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.cached_tokens += other.cached_tokens


@dataclass
class SearchAngle:
    description: str  # e.g. "national statistics office boundary data"
    channel_hint: str  # "catalog" | "web_search" | "national_geoportal"
    rationale: str
    # These four ARE the fetcher's parameters; an angle missing `url` is dropped.
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
    # The endpoint that serves the data, set only after an independent probe.
    resource_url: str | None = None
    # The trust split: `claim` is what the MODEL said, and not one key of it is verified.
    claim: dict = field(default_factory=dict)
    # What the HARNESS measured by re-probing; written by deterministic code.
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
    # Run-wide cost; "by_role" breaks the token counts down per stage.
    totals: dict = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    # Dead ends: excluded from candidates, recorded so nothing found vanishes silently.
    unresolved: list[Candidate] = field(default_factory=list)
    status: str = "ok"  # "ok" | "needs_human_review"
    reason: str | None = None  # populated when status is needs_human_review

    def to_dict(self) -> dict:
        return asdict(self)
