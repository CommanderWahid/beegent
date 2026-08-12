<div align="center">

# <img src="assets/beegent_logo.svg" alt="" height="72" valign="middle" /> Beegent

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: proof of concept](https://img.shields.io/badge/status-proof--of--concept-yellow)

**Open-source multi-agent pipeline that turns a `(country, use_case)` pair into a review-ready list of candidate geospatial data sources.**

</div>

A planner, catalog workers, a search & explore worker, and a triage/critic loop run as five
chained agents to find, evaluate, and hand off candidates — turning a day of manual searching
and tab-switching into a review-ready draft in minutes.

Accelerates the **technical scoping** step of your ingestion process:

```
business scoping -> [technical scoping: THIS TOOL] -> dev pipeline -> validation -> deployment
```

Today that step is manual: search the internet for a country/use-case's data, open a handful
of endpoints, compare them, explore columns, define a source-to-target mapping, and write up
metadata — before the dev pipeline work can even start. This is a working prototype of that
gap-closing step, not a polished system.

## Usage

```bash
uv sync
```

**Backend** defaults to Ollama running locally (`http://localhost:11434`). Pull the models
`core/config.py` expects:

```bash
ollama pull deepseek-r1:14b
ollama pull qwen3:8b
ollama pull llama3.1:8b
```

To use Databricks instead of Ollama:

1. Copy the template and fill in your workspace details:
   ```bash
   cp .env.example .env
   # then edit .env: set DATABRICKS_HOST and DATABRICKS_TOKEN
   ```
2. Set `LLM_BACKEND=databricks` — model defaults switch automatically to
   `databricks-claude-sonnet-4-6` / `databricks-claude-haiku-4-5` /
   `databricks-claude-haiku-4-5` / `databricks-claude-opus-5`
   (planner / search & explore / triage / critic). These must match serving-endpoint names in
   your workspace, not bare model names — check `GET /api/2.0/serving-endpoints` if you get a
   404. Override any of `PLANNER_MODEL`, `SEARCH_EXPLORE_MODEL`, `TRIAGE_MODEL`,
   `CRITIC_MODEL` in `.env` if you want a different model for a given step.

Example:

```bash
cp .env.example .env
# edit .env with your Databricks host + token

LLM_BACKEND=databricks uv run python core/run.py --country Kenya \
  --use-case "administrative boundaries for a flood-response dashboard"
```

Writes `candidate_list.json` (override with `--out`) and prints a run summary to stdout.
