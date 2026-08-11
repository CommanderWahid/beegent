"""Single source of truth for backend, models and tunables.

Everything else imports from here - no os.environ lookups and no literal model
names anywhere else in the codebase, so one file tells you what ran.
"""

import os

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" | "databricks"

# Model name is backend-specific: an Ollama model tag when LLM_BACKEND=ollama,
# or a Databricks serving-endpoint name when LLM_BACKEND=databricks.
#
# Local setup (RTX 4070 Laptop, 8GB)
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "deepseek-r1:14b")
SEARCH_EXPLORE_MODEL = os.environ.get("SEARCH_EXPLORE_MODEL", "qwen3:8b")
TRIAGE_MODEL = os.environ.get("TRIAGE_MODEL", "llama3.1:8b")  # small/cheap - once per candidate
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "deepseek-r1:14b")  # rare calls, highest stakes

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = 5  # cap on planner output, for cost control
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
