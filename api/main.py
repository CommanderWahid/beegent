"""FastAPI app: read the store, start a run, stream its log."""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from beegent import config
from beegent.llm import get_connector
from beegent.store import SqliteStore

from api.chat import GREETING, compose_use_case, interpret
from api.runner import Runner

app = FastAPI(title="beegent", docs_url="/api/docs", openapi_url="/api/openapi.json")
runner = Runner()


def _store() -> SqliteStore:
    """A real store or a clear 503 - an empty BEEGENT_DB is a configuration, not an error."""
    if not config.BEEGENT_DB:
        raise HTTPException(503, "no store configured: set BEEGENT_DB")
    return SqliteStore(config.BEEGENT_DB)


@app.get("/api/links")
def links(country: str = "") -> list[dict]:
    """Every stored link, newest row per URL, rot included.

    `claim` and `verification` are returned as stored and never merged: one is what a model
    said, the other is what the harness measured off the wire, and they may disagree.
    """
    return _store().catalog(country)


@app.get("/api/config")
def resolved_config() -> dict:
    """Which backend and models would answer - the CLI's [config] line, as JSON. No secrets."""
    c = get_connector()
    return {"backend": c.provider, "models": {r: c.model_for(r) for r in c.ROLES}}


@app.get("/api/countries")
def countries() -> list[dict]:
    """What the catalog holds, for the sidebar - derived, so it cannot drift from /api/links."""
    counts: dict[str, dict] = {}
    for row in _store().catalog():
        seen = counts.setdefault(row["country"], {"country": row["country"], "links": 0, "ok": 0})
        seen["links"] += 1
        seen["ok"] += row["status"] == "ok"
    return sorted(counts.values(), key=lambda c: c["country"])


@app.get("/api/runs")
def runs(country: str = Query(...), use_case: str = "", limit: int = 20) -> list[dict]:
    """What earlier runs tried, and what the critic said about them."""
    return _store().prior_runs(country, limit, use_case)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/api/chat")
def greeting() -> dict:
    """The agent introduces itself from ONE place, so the UI cannot drift from it."""
    return {"reply": GREETING}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """Interpret a message. A catalog question is answered with STORE ROWS, never prose.

    Narrating the catalog through a model would flatten `claim` (what a portal advertised)
    into `verification` (what the server sent), which is the one distinction that matters.
    """
    out = interpret(req.message, req.history)
    if out["intent"] == "catalog":
        out["links"] = _store().catalog(out.get("country", ""))
    return out


class RunRequest(BaseModel):
    country: str
    use_case: str
    # Collected by the chat and folded into use_case: discover() takes nothing else.
    format: str = ""
    vintage: str = ""
    channel: str = ""


@app.post("/api/runs", status_code=202)
def start_run(req: RunRequest) -> dict:
    """202 with an id, or 409: two runs would interleave writes to the one store."""
    if not (req.country.strip() and req.use_case.strip()):
        raise HTTPException(422, "country and use_case are both required")
    run_id = runner.start(req.country.strip(), compose_use_case(req.model_dump()))
    if run_id is None:
        raise HTTPException(409, "a run is already in flight")
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: str) -> dict:
    """Cancels for real: the pipeline checks between angles and agent steps."""
    if not runner.stop(run_id):
        raise HTTPException(404, "no such run")
    return {"stopping": run_id}


@app.get("/api/runs/{run_id}")
def run_result(run_id: str) -> dict:
    """The finished run, or 202 while it is still going."""
    result = runner.result(run_id)
    if result is None:
        raise HTTPException(202, "still running")
    return result


@app.get("/api/runs/{run_id}/logs")
def run_logs(run_id: str) -> StreamingResponse:
    def events():
        for line in runner.stream(run_id):
            # A log line may contain newlines; SSE needs one "data:" per line.
            yield "".join(f"data: {part}\n" for part in line.split("\n")) + "\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # or a proxy buffers the whole run into one burst
    })


#: Mounted LAST so every /api route wins; absent in dev, where `ng serve` proxies instead.
_UI = Path(__file__).resolve().parent.parent / "ui" / "dist" / "ui" / "browser"
if _UI.is_dir():
    app.mount("/", StaticFiles(directory=_UI, html=True), name="ui")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    # Fail before serving, exactly as run.py does: a bad endpoint otherwise hangs the chat.
    connector = get_connector()
    problem = connector.validate()
    if problem:
        raise SystemExit(f"error: {problem}")
    logging.info(
        f"[config] backend={connector.provider}  "
        + "  ".join(f"{r}={connector.model_for(r)}" for r in connector.ROLES)
    )
    # Localhost only, deliberately: this API spends real money on LLM calls.
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
