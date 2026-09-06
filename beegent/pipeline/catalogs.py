"""Deterministic catalog lookups - the non-agentic route to candidates."""

import logging

from beegent import config
from beegent.embedding import dot, from_blob, make_embedder
from beegent.schemas import Candidate
from beegent.store import NullStore, Store
from beegent.web_tools import WebTools

_log = logging.getLogger(__name__)


def query_catalogs(
    country: str,
    use_case: str,
    store: Store | None = None,
    tools: WebTools | None = None,
    embed=None,
) -> list[Candidate]:
    """Links a previous run verified, RE-PROBED now so nothing enters on an old promise."""
    store = store or NullStore()
    hits = store.verified_links(country)
    if not hits:
        return []

    embed = embed if embed is not None else make_embedder()
    if embed is None:  # no model, no matching - the pipeline degrades, it does not fail
        return []

    # A vector from another model is not comparable, and comparing anyway is silent nonsense.
    usable = [h for h in hits
              if h.get("embedding") and h.get("embed_model") == getattr(embed, "name", None)]
    if not usable:
        _log.info(f"  [catalog] {len(hits)} stored link(s), none embedded with {embed.name!r}")
        return []

    try:
        query_vec = embed([use_case])[0]
    except Exception as exc:
        _log.info(f"  [catalog] embedding failed: {type(exc).__name__}: {exc}")
        return []

    scored = sorted(((dot(query_vec, from_blob(h["embedding"])), h) for h in usable),
                    key=lambda pair: pair[0], reverse=True)
    relevant = [(score, h) for score, h in scored if score >= config.CATALOG_MIN_RELEVANCE]
    _log.info(f"  [catalog] {len(relevant)}/{len(usable)} stored link(s) match this use case")

    tools = tools or WebTools(log=lambda m: None)
    out: list[Candidate] = []
    for score, hit in relevant[: config.MAX_CATALOG_PROBES]:
        try:
            probe = tools.probe_url(hit["resource_url"])
        except Exception as exc:  # a budget or transport failure must not kill the run
            _log.info(f"  [catalog] probe failed: {type(exc).__name__}: {exc}")
            break
        if not probe.get("ok"):
            # Rot: the only place it can be seen is a re-probe, so record it here.
            _log.info(f"  [catalog] link no longer valid: {hit['resource_url']}")
            store.mark_link(hit["link_uid"], "ko")
            continue
        _log.info(f"  [catalog] {score:.3f} {hit['dataset'][:60]}")
        out.append(Candidate(
            url=hit["url"],
            title=hit["dataset"],
            dataset=hit["dataset"],
            source="catalog",
            confidence=hit.get("confidence"),
            resource_url=hit["resource_url"],
            claim=hit.get("claim") or {},
            verified_at=hit.get("last_verified", ""),
            verification=probe,  # THIS run's probe, never the stored one
        ))
    return out
