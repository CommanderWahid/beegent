"""
Single source of truth for backend, models and tunables.
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
        "GEOFETCH_MODEL": "qwen3:8b",  # qwen3 has the most reliable tool calling under
                                       # Ollama; deepseek-r1 does not. 8b (~5.2GB) fits an
                                       # 8GB card with room left for a 16k context. Do not
                                       # "upgrade" this to qwen3:14b - it was tried and
                                       # measured at ~10GB resident, 39% spilled to CPU,
                                       # and it blew past LLM_TIMEOUT on a single call.
        "CRITIC_MODEL": "deepseek-r1:14b",
    },
    "databricks": {  # serving-endpoint names, not bare model names - verify with
        # `GET /api/2.0/serving-endpoints` if these ever 404 in your workspace
        "PLANNER_MODEL": "databricks-claude-sonnet-4-6",
        "GEOFETCH_MODEL": "databricks-claude-haiku-4-5",
        "CRITIC_MODEL": "databricks-claude-opus-5",
    },
}
_defaults = _DEFAULT_MODELS.get(LLM_BACKEND, _DEFAULT_MODELS["ollama"])

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", _defaults["PLANNER_MODEL"])
GEOFETCH_MODEL = os.environ.get("GEOFETCH_MODEL", _defaults["GEOFETCH_MODEL"])  # tool-calling
# agent, once per angle - needs native function calling, this is the expensive one
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", _defaults["CRITIC_MODEL"])  # rare calls, highest stakes

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = 3  # cap on planner output, and the main cost lever: every angle is a
                # full agent run (GEOFETCH_MAX_STEPS calls), not one cheap search
MAX_ITERATIONS = 2  # hard cap on planner attempts (critic re-plans)

# --- geofetch agent (core/pipeline/geofetch.py) -------------------------------

GEOFETCH_MAX_STEPS = 20  # agent iterations (= LLM calls) per angle; the cost cap
GEOFETCH_MAX_HTTP_REQUESTS = 50  # web requests per angle, across all three tools
GEOFETCH_MIN_EFFORT_REQUESTS = 5  # a failure report filed before this many requests is
                                  # bounced back once with a checklist of untried techniques.
                                  # Weak models give up long before they have tried the
                                  # methodology; this is the floor on genuine effort.

# History compaction. Ollama silently evicts the OLDEST messages on context overflow -
# i.e. the system prompt and the task itself - so the agent forgets what it was doing.
# Trimming old tool results in place is what keeps a long run on-goal.
KEEP_FULL_TOOL_RESULTS = 5  # most recent tool results kept at full length
TRIM_TOOL_TO = 500  # older ones truncated to this many chars
TRIM_ASSISTANT_TO = 800  # cap on a kept assistant message

# --- web tools (core/search_backends/web_tools.py) ----------------------------

MAX_BODY_BYTES = 60_000  # per fetched page - local-model context economy
MAX_TEXT_CHARS = 4_000  # extracted HTML text kept per page
MAX_LINKS = 80  # hyperlinks reported per page
MAX_URLS_FOUND = 60  # entries in fetch_page()'s flat urls_found list
PROBE_BYTES = 16  # enough for every magic signature we know

# Verified beats self-assessed: found=True means the harness independently probed the file
# and the magic bytes matched the requested format, so even a "low" self-report outranks
# anything an unverified candidate could have scored.
CONFIDENCE_BY_REPORT = {"high": 0.95, "medium": 0.8, "low": 0.7}
CONFIDENCE_DEFAULT = 0.7

# Output cap. Cannot bind within one iteration now that MAX_ANGLES is 3, but still can
# across iterations - _rank() sees carried + fresh.
MAX_FINAL_CANDIDATES = 3

CHAT_JSON_ATTEMPTS = 2  # a JSON-mode call that comes back unparseable gets one retry

HTTP_TIMEOUT = 20

# Careful with these two together: the OpenAI SDK retries 429/5xx/timeouts with exponential
# backoff, so the worst-case wall clock for ONE completion is (LLM_MAX_RETRIES + 1) x
# LLM_TIMEOUT - and angles run serially, so a run multiplies that again by MAX_ANGLES.
# At the defaults below that is 4 x 180s = 12 min per call, 36 min for a fully-stuck run.
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3  # one more than the SDK default, to ride out a rate-limit burst
