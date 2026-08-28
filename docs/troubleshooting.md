# Troubleshooting

## The run finds nothing

**Every angle dead-ends with "no verifiable download found".**

Look at the `url` of each angle in the plan output. If they are portal homepages or search pages,
the planner did not find a concrete entry point — that is the single biggest predictor of a failed
angle. Try a more specific `--use-case`, naming the producing agency or the dataset if you know it.

**All angles fail immediately with empty search results.**

The search chain is being bot-blocked. `web_search` reports `engines_tried` and a note when all
three engines fail. Wait, or run from a different network. An empty result is not evidence that no
data exists, and the agent is told so.

**The data is behind a "download" button.**

Beegent does not execute JavaScript. If a file only becomes reachable after a client-side
interaction, it is genuinely out of reach and is reported as an honest failure. See
[The geofetch agent](geofetch.md).

## The run is expensive

**Token counts in the hundreds of thousands.**

Check `totals.by_role` — geofetch normally dominates. Then check whether an angle burned its whole
step budget: dead ends carry their own `cost`. The usual cause is a bad start URL. See
[Performance](performance.md).

**A single angle hit a rate limit.**

Almost always one huge tool result being re-sent every step. Lower `TOOL_RESULT_MAX_CHARS` or
`KEEP_FULL_TOOL_RESULTS` — not `GEOFETCH_MAX_STEPS`.

## Model and backend problems

**`cannot reach Ollama at ... Is it running?`**

Start it with `ollama serve`. If it is on another host, set `OLLAMA_BASE_URL`.

**The agent wanders off, ignores the task, or repeats itself.**

Almost always context, not the model. Start Ollama with `OLLAMA_CONTEXT_LENGTH=16384` — on a small
context it silently drops the oldest messages, which are the system prompt and the task.

**404 from Databricks.**

The model name must be a *serving-endpoint* name in your workspace, not a bare model name. Check
`GET /api/2.0/serving-endpoints`.

**`databricks backend needs DATABRICKS_HOST`**

Copy `.env.example` to `.env` and fill it in. Validation runs before the run starts, on purpose.

**The model never calls tools, or writes JSON as text.**

It needs native function calling. Some small models lose the ability in long conversations;
Beegent parses those and executes them anyway, but a model that never emits a real tool call is
the wrong model for `geofetch`. `qwen3:8b` is the tested local default.

## Output problems

**A candidate I expected is missing.**

Check `unresolved`. If it is there, `claim.failure_reason` says why. If the URL was reported but
rejected, the run log shows `REPORT REJECTED` with the reason — most often the file was live but
the wrong format, or the URL never appeared in a tool result.

**`resource_url` is null on a candidate.**

It should not be: candidates always carry a verified `resource_url`, and anything without one is
under `unresolved`. If you see this, please [open an issue](https://github.com/CommanderWahid/beegent/issues).

**The reported file size does not match.**

`claim.file_size_bytes` is what a portal page advertised; `verification.total_size_bytes` is what
the server sent. They are allowed to disagree — only the second was measured.

## Development

**Tests fail on a fresh clone.**

They should not — the suite is fully offline. Run `uv sync` first. Beegent supports Python 3.10+
Python 3.10, 3.11 and 3.12 are all supported.

**Ruff complains about something unrelated to my change.**

There is no ruff configuration in the repo yet, so you have to select the rules explicitly:
`uvx ruff check --select F,ERA .`. A bare `ruff check .` enables a much broader default set and
reports around 40 findings that the project does not currently treat as errors.
