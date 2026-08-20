"""Discovery pipeline stages, in run order - see core/run.py:discover()."""

from .catalog_workers import run_catalog_workers
from .critic import needs_escalation, run_critic
from .geofetch import resolve_angle
from .planner import plan

__all__ = [
    "plan",
    "run_catalog_workers",
    "resolve_angle",
    "needs_escalation",
    "run_critic",
]
