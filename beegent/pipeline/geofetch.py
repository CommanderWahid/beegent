"""Geofetch: the per-angle worker."""

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from beegent import config
from beegent.llm import chat_tools, message_text, strip_think, to_message_dict
from beegent.schemas import Candidate, SearchAngle, TokenUsage
from beegent.web_tools import WebTools, _URL_RE, normalize_url

_log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a download-URL resolver agent for open geodata (or any open data).
Input: a starting portal/dataset page URL, a dataset description, a desired
file format, and a vintage ("latest" or a specific date/year/version).
Goal: return the direct download URL of that exact file, verified.

You know nothing about any specific portal. Work it out with your tools,
using this GENERIC methodology:

1. RECONNAISSANCE. fetch_page the start URL. Many portals are JavaScript
   apps whose raw HTML is an empty shell (the tool flags this). Then look
   for machine-readable back-ends instead of scraping rendered pages:
   <link> tags (alternate/atom/dcat), API paths referenced in the page,
   sitemap.xml, content negotiation (retry with accept="application/xml"
   or "application/json"), and redirects - legacy dataset URLs often
   redirect to the current catalogue, and the redirect target's domain
   frequently hosts the download API itself.
1b. SEARCH THE WEB EARLY. When the portal is opaque, web_search is usually
   the shortest path: query the dataset's HUMAN name (not its catalogue
   identifier) + "download" + the format, or + "API"; also try the
   country's own language ("telechargement", "descarga", ...). Zero
   results usually means the engine bot-blocked you, not that nothing
   exists - rephrase and mix with other techniques. Also: fetch_page a
   <script src> bundle from a JS shell and scan it for absolute API base
   URLs (they appear as string literals) - that is discovery, not
   invention.
2. FIND THE DOWNLOAD SERVICE. Catalogues (GeoNetwork/CKAN/DCAT/CSW/STAC/
   OGC-API and homegrown ones) usually separate metadata from a download
   service. Metadata records list "distribution", "transferOptions",
   "resources" or Atom links pointing to it. Prefer structured feeds
   (Atom/JSON) over HTML.
2b. KNOW THE PLATFORM CONVENTIONS. Most portals run standard catalogue
   software; when a dataset page /datasets/<slug> is a JS shell, its
   machine-readable twin is usually one convention away on the same host:
   udata -> /api/1/datasets/<slug>/ ; CKAN ->
   /api/3/action/package_show?id=<slug> ; GeoNetwork ->
   /geonetwork/srv/api/records/<uuid>/formatters/xml ; DCAT ->
   /catalog.rdf or /data.json ; STAC -> /stac or /collections. Trying
   these on a discovered host is systematic variation, not invention.
2c. MINE urls_found. Every fetch_page result includes urls_found - the
   complete list of absolute URLs in the body. When a page discusses
   downloads (news posts, documentation, metadata records), the concrete
   service and file URLs are in that list: follow them instead of
   summarizing the page's prose.
3. FILTER, DON'T SCRAPE. Download services usually accept query parameters
   (format=..., page=..., zone=...). Inspect one response to learn the
   parameter and pagination vocabulary (attributes like page/pagecount/
   totalentries or rel=next links), then filter server-side. Enumerate ALL
   pages before concluding a file does not exist.
4. VINTAGE. For "latest", collect every candidate edition and compare their
   explicit edition/publication dates. Do NOT assume file-naming conventions
   are stable across editions (element order changes between releases) and
   do NOT construct a newer-looking URL by editing a date inside an old one.
5. NEVER INVENT URLS. Every URL you fetch or report must come from
   (a) a fetched document, (b) a redirect, or (c) a query-parameter or
   pagination variation of an endpoint you discovered. If a guess is
   unavoidable, fetch it and treat a failure as disproof, not noise.
6. VERIFY BEFORE REPORTING. probe_url the candidate. Only report if the
   probe shows an HTTP 200/206 serving a payload whose type matches the
   requested format and whose size is plausible for the dataset. An HTML
   body at a download URL is an error page.
7. REPORT with report_result, citing the evidence chain (which page led to
   which). Only report failure after you have genuinely exhausted the
   techniques above (search, feeds, content negotiation, JS bundles,
   redirects) - a failure report after a couple of requests will be
   rejected. Be economical, but persistent.
"""

# Re-sent on EVERY call, so wording is a per-step tax: keep these terse.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "GET a URL. HTML comes back as text + links; XML/JSON raw. "
                       "Flags redirects and JS-app shells. ALWAYS mine urls_found: "
                       "every absolute URL in the body, where download services and "
                       "API bases usually appear.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "accept": {"type": "string", "description": "Accept header, for content "
                                                        "negotiation"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Web search. Returns titles + URLs. Use early on an opaque JS "
                       "portal: dataset name + 'download' + format.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "probe_url",
        "description": "Range-request a URL's first bytes: status, size, and payload "
                       "type from magic bytes. Use before reporting any download URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "report_result",
        "description": "Final answer. Only after a successful probe_url of download_url.",
        "parameters": {"type": "object", "properties": {
            "found": {"type": "boolean"},
            "download_url": {"type": "string"},
            "edition": {"type": "string", "description": "edition/version id"},
            "vintage_date": {"type": "string"},
            "file_size_bytes": {"type": "integer"},
            "checksum": {"type": "string", "description": "if the service publishes one"},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "ordered discovery chain"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "failure_reason": {"type": "string"}},
            "required": ["found"]}}},
]

TOOL_NAMES = {"fetch_page", "probe_url", "web_search", "report_result"}

# Requested format -> acceptable probe payloads. GeoParquet stays strict, deliberately.
_ARCHIVES = {"zip", "7z", "gzip"}
FORMAT_PAYLOADS = {
    "geoparquet": {"parquet"}, "parquet": {"parquet"},
    "shapefile": set(_ARCHIVES), "shp": set(_ARCHIVES),
    "geopackage": {"sqlite/geopackage"} | _ARCHIVES,
    "gpkg": {"sqlite/geopackage"} | _ARCHIVES,
    "geotiff": {"tiff/geotiff"} | _ARCHIVES, "tiff": {"tiff/geotiff"} | _ARCHIVES,
    "pdf": {"pdf"},
}
TEXT_FORMATS = {"csv", "json", "geojson", "xml", "gml", "kml", "wkt", "txt"}


@dataclass
class AgentResult:
    found: bool
    download_url: str = ""
    report: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    steps_used: int = 0
    transcript: list = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


class GeofetchAgent:
    def __init__(self, tools: WebTools, max_steps: int | None = None,
                 log: Callable[[str], None] = lambda m: None):
        self.tools = tools
        self.max_steps = config.GEOFETCH_MAX_STEPS if max_steps is None else max_steps
        self.log = log
        self._discovered: set[str] = set()
        self._last_probe: dict = {}
        self._reject_reason: str = ""
        self._failure_nudged: bool = False

    # -- plumbing ----------------------------------------------------------- #

    def _dispatch(self, name: str, args: dict) -> dict:
        try:
            if name == "fetch_page":
                return self.tools.fetch_page(args["url"], args.get("accept", ""))
            if name == "probe_url":
                return self.tools.probe_url(args["url"])
            if name == "web_search":
                return self.tools.web_search(args["query"])
            return {"error": f"unknown tool {name}"}
        except Exception as exc:  # a tool crash is information for the agent, not a kill
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _harvest(self, payload: str) -> None:
        """Record every URL a tool result contained, so provenance can be checked later."""
        for m in _URL_RE.finditer(payload):
            self._discovered.add(normalize_url(m.group(0).rstrip(".,;")))

    # -- main loop ---------------------------------------------------------- #

    def run(self, start_url: str, dataset: str, fmt: str,
            vintage: str = "latest") -> AgentResult:
        """Drive the agent to a verified download, or to an honest failure."""
        task = {"start_url": start_url, "dataset": dataset,
                "format": fmt, "vintage": vintage}
        self._task_json = json.dumps(task)
        self._task_format = fmt
        self._discovered.add(normalize_url(start_url))
        self._call_seen: dict[str, int] = {}  # exact call -> first step it ran at
        self._repeats = 0  # enough suppressed duplicates means the model is stuck
        user = "Resolve this download URL:\n" + json.dumps(task, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        result = AgentResult(found=False)

        for step in range(1, self.max_steps + 1):
            result.steps_used = step
            _compact_history(messages)
            try:
                msg, usage = chat_tools("geofetch", messages, TOOL_SCHEMAS)
            except Exception as exc:
                # The one thing here that can still raise; degrade with the cost so far.
                self.log(f"[step {step}] aborted: {type(exc).__name__}: {exc}")
                result.report = {
                    "found": False,
                    "failure_reason": f"agent aborted: {type(exc).__name__}: {exc}",
                }
                return result
            result.usage.add(usage)
            messages.append(to_message_dict(msg))
            content = message_text(msg)
            if content:
                visible = strip_think(content)
                if visible:
                    self.log(f"[step {step}] {visible[:300]}")

            tool_calls = list(getattr(msg, "tool_calls", None) or [])

            if tool_calls:
                calls = [(c.function.name, _args_of(c), c.id) for c in tool_calls]
            else:
                rescued = extract_inline_tool_call(content)
                if rescued is None:
                    messages.append({"role": "user", "content":
                                     "Continue using tools, or call report_result. "
                                     "REMINDER of the task: " + self._task_json})
                    continue
                # Template drift: run the plain-text call anyway, with call_id None.
                calls = [(rescued[0], rescued[1], None)]

            for name, args, call_id in calls:
                if self._run_call(step, name, args, messages, result, call_id):
                    return result

        result.report = {"found": False,
                         "failure_reason": f"step budget ({self.max_steps}) exhausted"}
        return result

    def _reply(self, call_id, content: str) -> dict:
        """One reply, in whichever shape the call arrived as."""
        if call_id is None:
            return {"role": "user", "content": content}
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def _run_call(self, step: int, name: str, args: dict, messages: list,
                  result: "AgentResult", call_id=None) -> bool:
        """Execute one call from either path; True means the run is over."""
        rescued = call_id is None
        self.log(f"[step {step}] rescued inline {name} call" if rescued
                 else f"[step {step}] -> {name}({json.dumps(args)[:200]})")
        entry = {"step": step, "tool": name, "args": args}
        if rescued:
            entry["rescued"] = True
        result.transcript.append(entry)

        if name == "report_result":
            if rescued:
                args = coerce_report_args(args)
            outcome = self._handle_report(args)
            if outcome is not None:  # accepted, or an honest give-up
                _finish(result, outcome, args)
                return True
            body = json.dumps({"error": self._reject_reason,  # rejected: bounce it back
                               "probe": self._last_probe,
                               "task_reminder": self._task_json})
            if rescued:
                body += " Also: emit REAL tool calls, not JSON as text."
            messages.append(self._reply(call_id, body))
            return False

        # Anti-loop: an identical call is answered from memory, costing no budget.
        call_key = name + ":" + json.dumps(args, sort_keys=True)
        first = self._call_seen.get(call_key)
        if first is not None:
            self._repeats += 1
            self.log(f"  (repeated call suppressed: {name}, "
                     f"{self._repeats}/{config.GEOFETCH_MAX_REPEATS})")
            if self._repeats >= config.GEOFETCH_MAX_REPEATS:
                # A stuck angle gets more expensive per step while producing nothing.
                self.log(f"[step {step}] aborted: not converging "
                         f"({self._repeats} repeated calls)")
                result.report = {
                    "found": False,
                    "failure_reason": f"gave up: {self._repeats} repeated tool calls, "
                                      "the agent was not converging",
                }
                return True
            messages.append(self._reply(call_id, json.dumps({
                "error": f"REPEATED CALL: you already ran this exact {name} call at step "
                         f"{first} and the result has not changed. Do something DIFFERENT: "
                         "follow a link or URL you already saw, try a platform API "
                         "convention (see methodology), or change the query.",
                "task_reminder": self._task_json})))
            return False
        self._call_seen[call_key] = step

        payload = json.dumps(self._dispatch(name, args))
        self._harvest(payload)  # harvest URLs from the FULL payload, then truncate
        content = _fit(payload)
        if rescued:
            content = (f"Result of your {name} call (you wrote it as plain text - emit "
                       "REAL tool calls next time): " + content)
        messages.append(self._reply(call_id, content))
        return False

    def _handle_report(self, args: dict) -> dict | None:
        """Trust-but-verify: re-probe reported URLs ourselves."""
        if not args.get("found"):
            if (self.tools.requests_made < config.GEOFETCH_MIN_EFFORT_REQUESTS
                    and not self._failure_nudged):
                self._failure_nudged = True
                self._last_probe = {}
                self._reject_reason = (
                    "FAILURE REJECTED: you gave up after only "
                    f"{self.tools.requests_made} HTTP request(s). Before failing you must "
                    "try: web_search (dataset name + 'download' + format), content "
                    "negotiation (accept='application/xml'), fetching a <script src> "
                    "bundle to find API base URLs, and following any legacy-URL "
                    "redirects. Keep going.")
                return None
            return {"found": False}  # genuine failure after real effort: accept it
        url = args.get("download_url") or args.get("url") or ""
        if not url:
            self._last_probe = {"error": "no download_url in report"}
            self._reject_reason = ("REPORT REJECTED: found=true but no download_url "
                                   "was given.")
            return None
        if normalize_url(url) not in self._discovered:
            self._last_probe = {}
            self._reject_reason = (
                "REPORT REJECTED: this download_url never appeared in any tool result - "
                "you invented it. Only report URLs you actually discovered (fetch or "
                "probe a candidate first).")
            return None
        probe = self.tools.probe_url(url)
        self._last_probe = probe
        status_ok = probe.get("status") in (200, 206)
        payload = probe.get("payload_type", "unknown")
        fmt_l = (self._task_format or "").lower()
        allowed = FORMAT_PAYLOADS.get(fmt_l)
        if allowed is not None:
            ok = status_ok  # a mismatch returns below, so reaching the end means it matched
            if status_ok and payload not in allowed:
                self._reject_reason = (
                    f"REPORT REJECTED: the file is '{payload}' but the task asks for "
                    f"{self._task_format} (expected {sorted(allowed)}). This is the wrong "
                    "file or the wrong dataset - find the requested format.")
                return None
        elif fmt_l in TEXT_FORMATS:
            ok = (status_ok
                  and payload in ("json-text", "xml/html-text", "unknown")
                  and "html" not in probe.get("content_type", ""))
        else:
            ok = probe.get("ok", False)
        if not ok:
            self._reject_reason = (
                "REPORT REJECTED: independent probe of download_url failed. Do not report "
                "unverifiable URLs - keep searching.")
            return None
        return {"found": True, "download_url": url, "verification": probe}


def _args_of(call) -> dict:
    """A native tool call's arguments, tolerating malformed JSON."""
    try:
        return json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def _finish(result: AgentResult, outcome: dict, args: dict) -> AgentResult:
    result.found = outcome["found"]
    result.download_url = outcome.get("download_url", "")
    result.report = args
    result.verification = outcome.get("verification", {})
    return result


def _fit(payload: str) -> str:
    """Cap one tool result before it enters the conversation."""
    if len(payload) <= config.TOOL_RESULT_MAX_CHARS:
        return payload
    return payload[: config.TOOL_RESULT_MAX_CHARS] + (
        " ...[result truncated - refetch a narrower path or a specific page for more]")


def _compact_history(messages: list) -> None:
    """Trim old tool results and long assistant messages in place to fit a small context."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[: -config.KEEP_FULL_TOOL_RESULTS or None]:
        c = messages[i].get("content", "")
        if len(c) > config.TRIM_TOOL_TO:
            messages[i]["content"] = c[: config.TRIM_TOOL_TO] + " ...[older result trimmed]"
    for m in messages:
        if m.get("role") == "assistant" and len(m.get("content") or "") > config.TRIM_ASSISTANT_TO:
            m["content"] = m["content"][: config.TRIM_ASSISTANT_TO] + " ...[trimmed]"


def extract_inline_tool_call(text: str) -> tuple | None:
    """Rescue parser for models that write a tool call as plain text instead of emitting it."""
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("<tool_call>", " ").replace("</tool_call>", " ")
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if not (isinstance(obj, dict) and obj.get("name") in TOOL_NAMES):
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if isinstance(args, dict):
            return obj["name"], args
    return None


def coerce_report_args(args: dict) -> dict:
    """Normalize a malformed report_result: url -> download_url, bare error -> found false."""
    out = dict(args)
    if not out.get("download_url") and out.get("url"):
        out["download_url"] = out["url"]  # common field-name slip
    if "found" in out:
        return out
    if out.get("download_url"):
        out["found"] = True
    else:
        out["found"] = False
        out.setdefault("failure_reason", str(args.get("error") or json.dumps(args))[:400])
    return out


# --- stage entry point ---


# A whitelist, never a passthrough: `report` is model-authored.
_CLAIM_KEYS = ("edition", "vintage_date", "file_size_bytes", "checksum",
               "confidence", "evidence", "failure_reason")


def _claim(report: dict) -> dict:
    """The model's own account of the file, structured."""
    claim = {k: report[k] for k in _CLAIM_KEYS if report.get(k) not in (None, "", [], {})}
    if "evidence" in claim and not isinstance(claim["evidence"], list):
        claim["evidence"] = [claim["evidence"]]
    return claim


def _cost(result: AgentResult, tools: WebTools) -> dict:
    """What the angle actually cost."""
    return {
        "steps_used": result.steps_used,
        "http_requests": tools.requests_made,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
    }


def resolve_angle(
    angle: SearchAngle, log: Callable[[str], None] = _log.info,
) -> tuple[Candidate | None, Candidate | None]:
    """Resolve one search angle into a verified candidate."""
    tools = WebTools(log=log)
    agent = GeofetchAgent(tools=tools, log=log)
    start_url = angle.url

    dataset = angle.dataset or angle.description
    # The four values the agent is actually started with, logged at the point of use.
    log(f"    [geofetch] task: url      = {start_url}")
    log(f"    [geofetch]       dataset  = {dataset!r}")
    log(f"    [geofetch]       format   = {angle.format!r}"
        f"{' (any - magic-byte check disarmed)' if not angle.format else ''}")
    log(f"    [geofetch]       vintage  = {angle.vintage!r}")

    result = agent.run(
        start_url=start_url,
        dataset=dataset,
        fmt=angle.format,
        vintage=angle.vintage,
    )
    report = result.report or {}
    log(
        f"    [geofetch] {'VERIFIED' if result.found else 'no result'} in "
        f"{result.steps_used} step(s), {tools.requests_made} request(s), "
        f"{result.usage.total_tokens:,} tokens"
    )
    if result.transcript:  # what it actually did, in order - the first thing you want on a
        # surprising result, and the only record of it once the conversation is discarded
        log("    [geofetch] path: "
            + " -> ".join(t["tool"] + ("*" if t.get("rescued") else "")
                          for t in result.transcript))

    if not result.found:
        # Nothing found must never vanish silently - record the dead end and its cost.
        claim = _claim(report)
        claim.setdefault("failure_reason", "no verifiable download found")
        return None, Candidate(
            url=start_url,
            title=angle.description or angle.url,
            source="geofetch",
            confidence=None,
            resource_url=None,
            claim=claim,
            cost=_cost(result, tools),
        )

    return (
        Candidate(
            url=start_url,  # the page we cite; resource_url is the endpoint that serves data
            title=angle.description or angle.url,  # the edition lives in claim.edition
            source="geofetch",
            confidence=config.CONFIDENCE_BY_REPORT.get(
                str(report.get("confidence", "")).lower(), config.CONFIDENCE_DEFAULT
            ),
            resource_url=result.download_url,
            claim=_claim(report),
            verification=result.verification,
            cost=_cost(result, tools),
        ),
        None,
    )
