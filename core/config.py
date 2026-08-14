"""Single source of truth for backend, models and tunables.

Everything else imports from here - no os.environ lookups and no literal model
names anywhere else in the codebase, so one file tells you what ran.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # picks up DATABRICKS_HOST / DATABRICKS_TOKEN etc. from a .env at repo root,
# without overriding anything already set in the real environment

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" | "databricks"

# Model name is backend-specific: an Ollama model tag when LLM_BACKEND=ollama,
# or a Databricks serving-endpoint name when LLM_BACKEND=databricks. Defaults below are
# picked per backend so flipping LLM_BACKEND alone is enough - no per-model env vars
# required unless you want to override one.
_DEFAULT_MODELS = {
    "ollama": {  # Local setup (RTX 4070 Laptop, 8GB)
        "PLANNER_MODEL": "deepseek-r1:14b",
        "SEARCH_EXPLORE_MODEL": "qwen3:8b",
        "TRIAGE_MODEL": "llama3.1:8b",
        "CRITIC_MODEL": "deepseek-r1:14b",
    },
    "databricks": {  # serving-endpoint names, not bare model names - verify with
        # `GET /api/2.0/serving-endpoints` if these ever 404 in your workspace
        "PLANNER_MODEL": "databricks-claude-sonnet-4-6",
        "SEARCH_EXPLORE_MODEL": "databricks-claude-haiku-4-5",
        "TRIAGE_MODEL": "databricks-claude-haiku-4-5",
        "CRITIC_MODEL": "databricks-claude-opus-5",
    },
}
_defaults = _DEFAULT_MODELS.get(LLM_BACKEND, _DEFAULT_MODELS["ollama"])

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", _defaults["PLANNER_MODEL"])
SEARCH_EXPLORE_MODEL = os.environ.get("SEARCH_EXPLORE_MODEL", _defaults["SEARCH_EXPLORE_MODEL"])
TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", _defaults["TRIAGE_MODEL"])  # small/cheap - once per candidate
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", _defaults["CRITIC_MODEL"])  # rare calls, highest stakes

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = 10  # cap on planner output, for cost control
MAX_ITERATIONS = 2  # hard cap on planner attempts (critic re-plans)

MAX_SEARCHES_PER_ANGLE = 1
MAX_FETCHES_PER_ANGLE = 3
MAX_TOOL_TURNS_PER_ANGLE = 8  # safety net so a chatty model can't spin forever

MIN_CANDIDATES = 2  # escalation gate: fewer than this -> call the critic
# Single-publisher results only escalate when they are also thin - several distinct
# high-confidence datasets from one publisher is a good outcome, not a failure.
SINGLE_PUBLISHER_MIN_CANDIDATES = 3
MAX_FINAL_CANDIDATES = 6
TRIAGE_CONFIDENCE_FLOOR = 0.5
# Ceiling for a candidate with no identified resource endpoint. Deliberately below
# TRIAGE_CONFIDENCE_FLOOR: a page nobody could find a download or API for has not been shown
# to serve data at all, so it is dropped rather than ranked low. Applied before the floor
# comparison in merge_triage.triage(), and only ever downward.
NO_RESOURCE_CONFIDENCE_CAP = 0.3

# Resource resolution is the only stage that spends HTTP requests per candidate rather than
# per angle, so it gets its own ceiling. Worst case is roughly two probes per deduped
# candidate; results are memoized per process, so carried candidates cost nothing to recheck.
# Observed ~56 probes for a 20-candidate run, so this needs real headroom: exhausting it makes
# later candidates look unfetchable, which triage then caps and drops. resolve_resources()
# warns when it runs out so that never happens silently.
MAX_RESOURCE_PROBES = 150
RESOURCE_PROBE_BYTES = 2048  # range size for the content-type check - enough to get headers back
# A link-heavy page can offer dozens of resource-shaped URLs; probing them all would burn the
# whole run's budget on one candidate. The extractor puts its best evidence first, so the
# answer is in the first few or not there at all.
MAX_RESOURCE_CANDIDATES_PER_PAGE = 5
# So one site can't flood the list - but not so tight that a country whose open data is
# concentrated on a single national portal loses its best datasets. data.gouv.fr hit the old
# cap of 3 and would have discarded IGN's BD TOPO. MAX_FINAL_CANDIDATES is the real output cap.
MAX_PER_DOMAIN = 6

CHAT_JSON_ATTEMPTS = 2  # a JSON-mode call that comes back unparseable gets one retry

HTTP_TIMEOUT = 20
LLM_TIMEOUT = 180
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
