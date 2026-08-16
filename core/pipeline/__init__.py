"""Discovery pipeline stages, in run order - see core/run.py:discover()."""

from .catalog_workers import run_catalog_workers
from .critic import needs_escalation, run_critic
from .merge_triage import merge_and_triage
from .planner import plan
from .search_explore import explore_angle

__all__ = [
    "plan",
    "run_catalog_workers",
    "explore_angle",
    "merge_and_triage",
    "needs_escalation",
    "run_critic",
]
