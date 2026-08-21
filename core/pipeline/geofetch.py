"""Geofetch: the per-angle worker. Resolves a search angle into a VERIFIED download URL.

One tool-calling agent per angle, given three deterministic tools (fetch_page, web_search,
probe_url) and a generic methodology - never a portal-specific parser. All the intelligence
is in the model; the tools are dumb on purpose.

What makes this different from finding a page that looks like data: the model can only
finish by calling report_result, and when it does, the harness INDEPENDENTLY re-probes the
reported URL with deterministic code before accepting it. Five guardrails police the known
failure modes:

  * provenance   - a URL that never appeared in a tool result is rejected as invented
  * re-probe     - bad status, or an HTML error page at a download URL, is rejected
  * format match - the probed file's magic bytes must match the requested format, so
                   drifting to the wrong dataset fails verification
  * min effort   - a failure report filed before GEOFETCH_MIN_EFFORT_REQUESTS requests is
                   bounced back once with a checklist of untried techniques
  * anti-loop    - an exactly repeated tool call is answered from memory with a warning
                   instead of re-spending HTTP budget

Plus two small-model survival measures: history compaction (Ollama silently evicts the
OLDEST messages on overflow - the system prompt and the task - so old tool results are
trimmed in place) and a rescue parser for template drift (long conversations make some
models write {"name": ..., "arguments": ...} as plain text instead of emitting a real tool
call; it is executed anyway, with a reminder).

A fabricated URL therefore cannot reach candidate_list.json. The worst case is an honest
failure, recorded in `unresolved`.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from core import config
from core.llm import chat_tools, message_text, strip_think, to_message_dict
from core.schemas import Candidate, SearchAngle, TokenUsage
from core.search_backends import WebTools, normalize_url
from core.search_backends.web_tools import _URL_RE  # private: not part of the package API

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

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "GET a URL. HTML is returned as extracted text + all "
                       "hyperlinks; XML/JSON is returned raw (truncated). "
                       "Reports redirects and JS-app shells. ALWAYS check "
                       "the urls_found list: it contains every absolute URL "
                       "in the body - download services, API bases and file "
                       "links usually appear there.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "accept": {"type": "string",
                       "description": "optional Accept header for content "
                                      "negotiation"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Web search. Returns result titles + URLs. Use early "
                       "when a portal is an opaque JS app: dataset name + "
                       "'download' + format/API.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "probe_url",
        "description": "Cheap range-request of a URL's first bytes: HTTP "
                       "status, total size, and payload type from magic "
                       "bytes. Use before reporting any download URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "report_result",
        "description": "Final answer. Only call after a successful probe_url "
                       "of download_url.",
        "parameters": {"type": "object", "properties": {
            "found": {"type": "boolean"},
            "download_url": {"type": "string"},
            "edition": {"type": "string",
                        "description": "edition/version identifier"},
            "vintage_date": {"type": "string"},
            "file_size_bytes": {"type": "integer"},
            "checksum": {"type": "string",
                         "description": "checksum if the service advertises one"},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "ordered discovery chain"},
            "confidence": {"type": "string",
                           "enum": ["high", "medium", "low"]},
            "failure_reason": {"type": "string"}},
            "required": ["found"]}}},
]

TOOL_NAMES = {"fetch_page", "probe_url", "web_search", "report_result"}

# Requested-format -> acceptable probe payload types (magic-byte classes).
# Bulk geodata is routinely shipped inside a container: IGN publishes national GeoPackage as
# split .7z.001 archives, Shapefile is almost always zipped, GeoTIFF is often gzipped. `zip`
# was already accepted here on exactly that reasoning; _ARCHIVES just makes the rule complete
# and consistent instead of accidentally zip-only.
#
# GeoParquet is deliberately NOT loosened: Parquet is not archive-shipped, and admitting
# containers there would let any zip satisfy the format most often asked for - which is the
# one place this guardrail earns its keep.
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
        """Drive the agent to a verified download, or to an honest failure.

        Never raises: a tool crash becomes a tool result the model can react to, and an LLM
        transport error (rate limit, timeout past the SDK's retries) becomes a failed
        AgentResult carrying the steps and tokens already spent. Callers get a result object
        in every case, so nothing an angle did can vanish from the run's accounting.
        """
        task = {"start_url": start_url, "dataset": dataset,
                "format": fmt, "vintage": vintage}
        self._task_json = json.dumps(task)
        self._task_format = fmt
        self._discovered.add(normalize_url(start_url))
        self._call_seen: dict[str, int] = {}  # exact call -> first step it ran at
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
                msg, usage = chat_tools(config.GEOFETCH_MODEL, messages, TOOL_SCHEMAS)
            except Exception as exc:
                # The LLM call is the one thing in this loop that can still raise - tool
                # crashes are already caught in _dispatch(). A rate limit or a timeout that
                # outlives the SDK's retries must degrade to an honest failure carrying the
                # cost spent so far, not blow the angle away with no record: the angle most
                # likely to be throttled is exactly the one you need to see in the output.
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

            if not tool_calls:
                rescued = extract_inline_tool_call(content)
                if rescued is None:
                    messages.append({"role": "user", "content":
                                     "Continue using tools, or call report_result. "
                                     "REMINDER of the task: " + self._task_json})
                    continue
                # Template drift: the model wrote a tool call as plain text. Execute it
                # anyway - the intent is unambiguous - and remind it to emit real calls.
                name, args = rescued
                self.log(f"[step {step}] rescued inline {name} call")
                result.transcript.append({"step": step, "tool": name,
                                          "args": args, "rescued": True})
                if name == "report_result":
                    args = coerce_report_args(args)
                    outcome = self._handle_report(args)
                    if outcome is not None:
                        return _finish(result, outcome, args)
                    messages.append({"role": "user", "content": json.dumps({
                        "error": self._reject_reason,
                        "probe": self._last_probe,
                        "task_reminder": self._task_json})
                        + " Also: emit REAL tool calls, not JSON as text."})
                    continue
                tool_result = self._dispatch(name, args)
                payload = json.dumps(tool_result)
                self._harvest(payload)
                messages.append({"role": "user", "content":
                                 f"Result of your {name} call (you wrote it as plain "
                                 "text - emit REAL tool calls next time): " + _fit(payload)})
                continue

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.log(f"[step {step}] -> {name}({json.dumps(args)[:200]})")
                result.transcript.append({"step": step, "tool": name, "args": args})

                if name == "report_result":
                    outcome = self._handle_report(args)
                    if outcome is not None:  # accepted, or an honest give-up
                        return _finish(result, outcome, args)
                    messages.append({  # rejected: bounce it back and keep searching
                        "role": "tool", "tool_call_id": call.id,
                        "content": json.dumps({
                            "error": self._reject_reason,
                            "probe": self._last_probe,
                            "task_reminder": self._task_json})})
                    continue

                # Anti-loop: an identical call is answered from memory with a warning
                # instead of re-spending HTTP budget and context on the same result.
                call_key = name + ":" + json.dumps(args, sort_keys=True)
                first = self._call_seen.get(call_key)
                if first is not None:
                    self.log(f"  (repeated call suppressed: {name})")
                    messages.append({
                        "role": "tool", "tool_call_id": call.id,
                        "content": json.dumps({
                            "error": f"REPEATED CALL: you already ran this exact {name} "
                                     f"call at step {first} and the result has not "
                                     "changed. Do something DIFFERENT: follow a link or "
                                     "URL you already saw, try a platform API convention "
                                     "(see methodology), or change the query.",
                            "task_reminder": self._task_json})})
                    continue
                self._call_seen[call_key] = step

                tool_result = self._dispatch(name, args)
                payload = json.dumps(tool_result)
                self._harvest(payload)  # harvest URLs from the FULL payload, then truncate
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": _fit(payload)})

        result.report = {"found": False,
                         "failure_reason": f"step budget ({self.max_steps}) exhausted"}
        return result

    def _handle_report(self, args: dict) -> dict | None:
        """Trust-but-verify: re-probe reported URLs ourselves.
        Returns the accepted outcome, or None to reject the report and bounce it back."""
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
            ok = status_ok and payload in allowed
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


def _finish(result: AgentResult, outcome: dict, args: dict) -> AgentResult:
    result.found = outcome["found"]
    result.download_url = outcome.get("download_url", "")
    result.report = args
    result.verification = outcome.get("verification", {})
    return result



def _fit(payload: str) -> str:
    """Cap one tool result before it enters the conversation.

    Every kept result is resent on every subsequent step, so an uncapped one is not a one-off
    cost - it is a per-step tax for the rest of the angle. A single 60KB WFS GetCapabilities
    once drove one angle to 861k input tokens and tripped a workspace rate limit, starving
    every angle after it. The marker matters: the model has to know it was cut so it can
    re-fetch something narrower instead of concluding the document ends there.
    """
    if len(payload) <= config.TOOL_RESULT_MAX_CHARS:
        return payload
    return payload[: config.TOOL_RESULT_MAX_CHARS] + (
        " ...[result truncated - refetch a narrower path or a specific page for more]")


def _compact_history(messages: list) -> None:
    """Trim old tool results and long assistant rambles IN PLACE so the conversation fits
    small local context windows. Without this, Ollama silently evicts the OLDEST messages
    on overflow - i.e. the system prompt and the task itself - and the model drifts
    off-goal, which looks like a model failure but is a context failure."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[: -config.KEEP_FULL_TOOL_RESULTS or None]:
        c = messages[i].get("content", "")
        if len(c) > config.TRIM_TOOL_TO:
            messages[i]["content"] = c[: config.TRIM_TOOL_TO] + " ...[older result trimmed]"
    for m in messages:
        if m.get("role") == "assistant" and len(m.get("content") or "") > config.TRIM_ASSISTANT_TO:
            m["content"] = m["content"][: config.TRIM_ASSISTANT_TO] + " ...[trimmed]"


def extract_inline_tool_call(text: str) -> tuple | None:
    """Rescue parser for local-model template drift: long conversations make some models
    (qwen3 among them) stop emitting native tool calls and write
    {"name": ..., "arguments": {...}} as plain text instead. Find and parse such a call
    anywhere in the content. Returns (name, args) or None."""
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
    """Normalize a malformed report_result (e.g. {"error": ...} with no 'found' field)
    into the expected schema. Weak models misname fields; that is not a reason to lose
    a genuine result."""
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


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


# What the model is allowed to contribute to Candidate.claim. A whitelist, not a passthrough:
# `report` is model-authored, so copying it wholesale would let a model put arbitrary keys -
# including something that reads like a verification result - into the output. `found` and
# `download_url` are excluded as duplicates of list membership and `resource_url`.
_CLAIM_KEYS = ("edition", "vintage_date", "file_size_bytes", "checksum",
               "confidence", "evidence", "failure_reason")


def _claim(report: dict) -> dict:
    """The model's own account of the file, structured. Never verified - see Candidate.

    Shape is normalized, truth is not: `evidence` is guaranteed to be a list so a consumer
    can always iterate it, because a model that returns it as a bare string would otherwise
    have callers walking it one character at a time. The contents stay exactly as claimed.
    """
    claim = {k: report[k] for k in _CLAIM_KEYS if report.get(k) not in (None, "", [], {})}
    if "evidence" in claim and not isinstance(claim["evidence"], list):
        claim["evidence"] = [claim["evidence"]]
    return claim


def _cost(result: AgentResult, tools: WebTools) -> dict:
    """What the angle actually cost. Measured by the harness, not reported by the model.

    Recorded per angle rather than only per run because the expensive thing to spot is a
    single dead end that burned the whole step budget.
    """
    return {
        "steps_used": result.steps_used,
        "http_requests": tools.requests_made,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
    }


def resolve_angle(
    angle: SearchAngle, log: Callable[[str], None] = print,
) -> tuple[Candidate | None, Candidate | None]:
    """Resolve one search angle into a verified candidate.

    Returns (verified, unresolved) - at most one is non-None. `verified` always carries a
    probed `resource_url` and a `verification` block; `unresolved` records an angle that
    honestly found nothing, so a dead end is visible in the output rather than silent.

    Both slots carry `claim` (what the model said) and `cost` (what the angle actually
    spent). Only `verified` carries `verification` - there is nothing to verify on a miss.
    """
    tools = WebTools(log=log)
    agent = GeofetchAgent(tools=tools, log=log)
    start_url = angle.url

    dataset = angle.dataset or angle.description
    # The four values the agent is actually started with - log them together, at the point
    # of use, so a surprising result can be read back against the task that produced it
    # rather than reconstructed from the planner's output three stages upstream.
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
        # Nothing found must never vanish silently: record the dead end, with the page we
        # did look at, the reason the agent gave up, and what the attempt cost.
        claim = _claim(report)
        claim.setdefault("failure_reason", "no verifiable download found")
        return None, Candidate(
            url=start_url,
            title=angle.url,
            source="geofetch",
            confidence=None,
            resource_url=None,
            claim=claim,
            cost=_cost(result, tools),
        )

    return (
        Candidate(
            url=start_url,  # the page we cite; resource_url is the endpoint that serves data
            title=angle.url,  # what we cite; the edition lives in claim.edition
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
