"""Backend-independent pipeline tunables, and the backend selector."""

import os

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # key into connectors.CONNECTORS
# Cross-run memory, on by default. Expanded HERE so a "~" set in .env - where no shell
# expansion happens - does not become a literal "~" directory. Set empty to disable.
BEEGENT_DB = os.path.expanduser(os.environ.get("BEEGENT_DB", "~/.beegent/memory.db"))
# Catalog matching model. Empty disables embeddings entirely; vectors are only ever
# comparable within one model, so the name is stored beside every vector.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _int(name: str, default: int) -> int:
    """An env var of the same name wins; a bad value fails at import, not mid-run."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"error: {name}={raw!r} is not an integer")
    if value < 1:
        raise SystemExit(f"error: {name}={value} must be at least 1")
    return value


# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = _int("MAX_ANGLES", 3)  # main cost lever: every angle is a full agent run
MAX_CATALOG_PROBES = _int("MAX_CATALOG_PROBES", 3)  # stored links re-probed per run
# Within this, a stored link answers the run outright and no LLM is called at all.
# It is the ONLY thing forcing periodic rediscovery, so 0 disables the short-circuit.
CATALOG_FRESH_DAYS = _int("CATALOG_FRESH_DAYS", 7)
MEMORY_RUNS = _int("MEMORY_RUNS", 3)  # past runs replayed to the planner, when a store exists
MAX_ITERATIONS = _int("MAX_ITERATIONS", 2)  # hard cap on planner attempts (critic re-plans)

# --- geofetch agent (beegent/pipeline/geofetch.py) ---------------------------

GEOFETCH_MAX_STEPS = _int("GEOFETCH_MAX_STEPS", 20)  # LLM calls per angle; the cost cap
GEOFETCH_MAX_HTTP_REQUESTS = _int("GEOFETCH_MAX_HTTP_REQUESTS", 50)  # per angle, all tools
GEOFETCH_MAX_REPEATS = _int("GEOFETCH_MAX_REPEATS", 3)  # identical calls before abandoning
GEOFETCH_MIN_EFFORT_REQUESTS = _int("GEOFETCH_MIN_EFFORT_REQUESTS", 5)  # floor before a give-up

# History compaction: Ollama evicts the oldest messages on overflow, losing the task itself.
KEEP_FULL_TOOL_RESULTS = _int("KEEP_FULL_TOOL_RESULTS", 3)  # recent results kept full length
TRIM_TOOL_TO = _int("TRIM_TOOL_TO", 500)  # older ones truncated to this many chars
TOOL_RESULT_MAX_CHARS = _int("TOOL_RESULT_MAX_CHARS", 8_000)  # the real input-token lever
TRIM_ASSISTANT_TO = _int("TRIM_ASSISTANT_TO", 800)  # cap on a kept assistant message

# --- web tools (beegent/web_tools.py) ----------------------------------------

MAX_BODY_BYTES = _int("MAX_BODY_BYTES", 20_000)  # raw XML/JSON body kept per fetched page
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 2_000)  # HTML text kept; the agent navigates by links
MAX_LINKS = _int("MAX_LINKS", 80)  # hyperlinks reported per page
MAX_URLS_FOUND = _int("MAX_URLS_FOUND", 60)  # entries in fetch_page()'s flat urls_found list
PROBE_BYTES = 16  # NOT overridable: a correctness floor, the geopackage signature is 16 bytes
PROBE_TEXT_BYTES = _int("PROBE_TEXT_BYTES", 65_536)  # second read when the payload is JSON/XML

# Verified beats self-assessed, so even a "low" self-report outranks anything unverified.
# Not overridable: a dict and its float default, with no per-run reason to retune them.
# Cosine a stored dataset must reach to be offered. MEASURED, not guessed - see
# tools/calibrate_relevance.py, and re-measure whenever EMBED_MODEL changes. Not
# overridable: a float, which _int() neither handles nor needs.
CATALOG_MIN_RELEVANCE = 0.30

CONFIDENCE_BY_REPORT = {"high": 0.95, "medium": 0.8, "low": 0.7}
CONFIDENCE_DEFAULT = 0.7

MAX_FINAL_CANDIDATES = _int("MAX_FINAL_CANDIDATES", 3)  # output cap; binds across iterations

CHAT_JSON_ATTEMPTS = _int("CHAT_JSON_ATTEMPTS", 2)  # an unparseable JSON-mode call gets one retry

HTTP_TIMEOUT = _int("HTTP_TIMEOUT", 20)

# Worst case for ONE completion is (LLM_MAX_RETRIES + 1) x LLM_TIMEOUT, and angles run serially.
LLM_TIMEOUT = _int("LLM_TIMEOUT", 180)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 3)  # one more than the SDK default, for rate-limit bursts
