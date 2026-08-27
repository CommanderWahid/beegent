"""Backend-independent pipeline tunables, and the backend selector."""

import os

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # key into connectors.CONNECTORS

# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = 3  # main cost lever: every angle is a full agent run
MAX_ITERATIONS = 2  # hard cap on planner attempts (critic re-plans)

# --- geofetch agent (beegent/pipeline/geofetch.py) ---------------------------

GEOFETCH_MAX_STEPS = 20  # agent iterations (= LLM calls) per angle; the cost cap
GEOFETCH_MAX_HTTP_REQUESTS = 50  # web requests per angle, across all three tools
GEOFETCH_MAX_REPEATS = 3  # identical calls refused before the angle is abandoned
GEOFETCH_MIN_EFFORT_REQUESTS = 5  # floor on effort before a failure report is accepted

# History compaction: Ollama evicts the oldest messages on overflow, losing the task itself.
KEEP_FULL_TOOL_RESULTS = 3  # most recent tool results kept at full length
TRIM_TOOL_TO = 500  # older ones truncated to this many chars
TOOL_RESULT_MAX_CHARS = 8_000  # the real lever on input tokens - every kept result is resent
TRIM_ASSISTANT_TO = 800  # cap on a kept assistant message

# --- web tools (beegent/web_tools.py) ----------------------------------------

MAX_BODY_BYTES = 20_000  # raw XML/JSON body kept per fetched page
MAX_TEXT_CHARS = 2_000  # extracted HTML text kept per page; the agent navigates by links
MAX_LINKS = 80  # hyperlinks reported per page
MAX_URLS_FOUND = 60  # entries in fetch_page()'s flat urls_found list
PROBE_BYTES = 16  # enough for every magic signature we know

# Verified beats self-assessed, so even a "low" self-report outranks anything unverified.
CONFIDENCE_BY_REPORT = {"high": 0.95, "medium": 0.8, "low": 0.7}
CONFIDENCE_DEFAULT = 0.7

MAX_FINAL_CANDIDATES = 3  # output cap; binds across iterations, not within one

CHAT_JSON_ATTEMPTS = 2  # a JSON-mode call that comes back unparseable gets one retry

HTTP_TIMEOUT = 20

# Worst case for ONE completion is (LLM_MAX_RETRIES + 1) x LLM_TIMEOUT, and angles run serially.
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3  # one more than the SDK default, to ride out a rate-limit burst
