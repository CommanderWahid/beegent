"""
Backend-independent pipeline tunables, and the backend selector.

What is NOT here: anything specific to one backend. Model names, base URLs and credentials
live on the connector that owns them (beegent/connectors/), so adding a backend never edits
this file. `run.py` still prints the resolved models and DiscoveryRun.models records them,
so "read one place to see what ran" still holds - it is just the run output rather than this
module.
"""

import os

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # a key into
# beegent.connectors.CONNECTORS - the registry is the source of truth for what is valid,
# including any connector registered at runtime

# --- pipeline tunables -------------------------------------------------------

MAX_ANGLES = 3  # cap on planner output, and the main cost lever: every angle is a
                # full agent run (GEOFETCH_MAX_STEPS calls), not one cheap search
MAX_ITERATIONS = 2  # hard cap on planner attempts (critic re-plans)

# --- geofetch agent (beegent/pipeline/geofetch.py) -------------------------------

GEOFETCH_MAX_STEPS = 20  # agent iterations (= LLM calls) per angle; the cost cap
GEOFETCH_MAX_HTTP_REQUESTS = 50  # web requests per angle, across all three tools
GEOFETCH_MAX_REPEATS = 3  # exact tool calls refused by the anti-loop guard before the angle
                          # is abandoned. A model that has had three identical calls bounced
                          # is not converging, and every further step re-sends the whole
                          # conversation: an observed run hit 7 suppressions and 0 attempts
                          # to finish, burning ~180k tokens to produce nothing.
GEOFETCH_MIN_EFFORT_REQUESTS = 5  # a failure report filed before this many requests is
                                  # bounced back once with a checklist of untried techniques.
                                  # Weak models give up long before they have tried the
                                  # methodology; this is the floor on genuine effort.

# History compaction. Ollama silently evicts the OLDEST messages on context overflow -
# i.e. the system prompt and the task itself - so the agent forgets what it was doing.
# Trimming old tool results in place is what keeps a long run on-goal.
KEEP_FULL_TOOL_RESULTS = 3  # most recent tool results kept at full length
TRIM_TOOL_TO = 500  # older ones truncated to this many chars
# Hard cap on a tool result entering the conversation at all. THIS is the lever on input
# tokens, not GEOFETCH_MAX_STEPS: every kept result is resent on every step, so one
# 60KB WFS GetCapabilities held at full length costs ~15k tokens per step for the rest of
# the run. An uncapped result once burned an 860k-token workspace minute on a single angle.
TOOL_RESULT_MAX_CHARS = 8_000
TRIM_ASSISTANT_TO = 800  # cap on a kept assistant message

# --- web tools (beegent/web_tools.py) -----------------------------------------

MAX_BODY_BYTES = 20_000  # raw XML/JSON body kept per fetched page. Capabilities
                         # documents run far larger and are not more useful for it:
                         # the agent is looking for a layer name or a link
MAX_TEXT_CHARS = 2_000  # extracted HTML text kept per page. The agent navigates by
                        # LINKS, not prose - the run that verified a download read almost
                        # no page text - and nothing downstream consumes this any more
                        # (Candidate.description became structured claim/verification).
                        # Halved rather than dropped: prose still catches inline format
                        # and coverage mentions that never became a link.
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
# For token-per-minute limits the lever is TOOL_RESULT_MAX_CHARS / KEEP_FULL_TOOL_RESULTS
# above, not the step cap: input tokens are dominated by tool results resent every step.
# At the defaults below that is 4 x 180s = 12 min per call, 36 min for a fully-stuck run.
LLM_TIMEOUT = 180
LLM_MAX_RETRIES = 3  # one more than the SDK default, to ride out a rate-limit burst
