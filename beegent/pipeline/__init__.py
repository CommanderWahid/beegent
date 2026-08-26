"""Discovery pipeline stages, in run order - see beegent/run.py:discover()."""

from .catalogs import query_catalogs
from .critic import needs_escalation, run_critic
from .geofetch import resolve_angle
from .planner import plan

__all__ = [
    "plan",
    "query_catalogs",
    "resolve_angle",
    "needs_escalation",
    "run_critic",
]
