#!/usr/bin/env python3
"""The catalog stage: stored links, matched by meaning and re-probed now rather than trusted."""

from beegent import config
from beegent.embedding import normalise
from beegent.pipeline.catalogs import query_catalogs
from beegent.schemas import Candidate, DiscoveryRun
from beegent.store import NullStore, SqliteStore

from tests.conftest import FILE_URL, PORTAL, build_tools

BOUNDARIES = "administrative boundaries from an OGC API-Features or WFS endpoint"
BUILDINGS = "outlines of houses and structures"


def _embedder(name="fake-model"):
    """Two themed axes, so a boundaries query and a buildings dataset score ~0 by design."""
    def embed(texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append(normalise([
                1.0 if ("boundar" in low or "limits" in low) else 0.0,
                1.0 if ("building" in low or "footprint" in low or "houses" in low) else 0.0,
                0.01,  # keeps an unrelated string from being a zero vector
            ]))
        return out
    embed.name = name
    return embed


_DEFAULT = object()  # so embed=None can mean "genuinely no embedder", not "use the default"


def _store_with(tmp_path, *datasets, resource=FILE_URL, embed=_DEFAULT):
    store = SqliteStore(str(tmp_path / "m.db"),
                        embed=_embedder() if embed is _DEFAULT else embed)
    cands = [Candidate(url=PORTAL, title=d, dataset=d, source="geofetch", confidence=0.95,
                       resource_url=(resource if i == 0 else f"{resource}?n={i}"),
                       claim={"edition": "2025"}, verification={"ok": True})
             for i, d in enumerate(datasets)]
    store.record_run(DiscoveryRun(country="Atlantis", use_case="x", candidates=cands), [])
    return store


def test_no_store_means_no_hits():
    """The default is a NullStore, so behaviour is unchanged unless memory is switched on."""
    assert query_catalogs("Atlantis", BOUNDARIES, NullStore()) == []


def test_no_embedder_means_no_matching(tmp_path):
    """Without a model the pipeline degrades to plain discovery; it must not fail."""
    store = _store_with(tmp_path, "administrative boundaries, whole country")
    assert query_catalogs("Atlantis", BOUNDARIES, store, build_tools(), embed=None) == []


def test_a_matching_link_comes_back_verified_by_THIS_run(tmp_path):
    """The guarantee: the verification block must be the fresh probe, not the stored one."""
    store = _store_with(tmp_path, "administrative boundaries, whole country")
    out = query_catalogs("Atlantis", BOUNDARIES, store, build_tools(), _embedder())
    assert len(out) == 1
    assert out[0].source == "catalog"
    assert out[0].verification["payload_type"] == "parquet", "a probe run just now"
    assert out[0].verification["status"] == 206


def test_a_different_theme_is_not_returned(tmp_path):
    store = _store_with(tmp_path, "BAG building footprints, whole country")
    assert query_catalogs("Atlantis", BOUNDARIES, store, build_tools(), _embedder()) == []


def test_matching_is_by_MEANING_not_shared_words(tmp_path):
    """The reason embeddings replaced word overlap: no token here is shared with the stored text."""
    store = _store_with(tmp_path, "administrative boundaries, whole country")
    out = query_catalogs("Atlantis", "municipal limits and provincial borders",
                         store, build_tools(), _embedder())
    assert len(out) == 1


def test_a_vector_from_another_model_is_excluded_not_compared(tmp_path):
    """Mixing vector spaces yields confident nonsense with no error, so it must never happen."""
    store = _store_with(tmp_path, "administrative boundaries, whole country",
                        embed=_embedder("model-a"))
    assert query_catalogs("Atlantis", BOUNDARIES, store, build_tools(),
                          _embedder("model-b")) == []


def test_a_link_with_no_vector_is_invisible(tmp_path):
    """Which is why the backfill is not optional once matching is embedding-only."""
    store = _store_with(tmp_path, "administrative boundaries, whole country", embed=None)
    assert query_catalogs("Atlantis", BOUNDARIES, store, build_tools(), _embedder()) == []


def test_a_dead_link_is_marked_ko_and_withheld(tmp_path):
    """Rot is only visible on a re-probe, so that is where it must be recorded."""
    store = _store_with(tmp_path, "administrative boundaries, whole country",
                        resource="https://api.atlantis.example/gone.parquet")
    assert query_catalogs("Atlantis", BOUNDARIES, store, build_tools(), _embedder()) == []
    assert store.verified_links("Atlantis") == [], "the rotted link must not be offered again"


def test_the_probe_count_is_capped(monkeypatch, tmp_path):
    """Without a cap a well-covered country costs one request per stored link, every run."""
    monkeypatch.setattr(config, "MAX_CATALOG_PROBES", 1)
    store = _store_with(tmp_path, "administrative boundaries north",
                        "administrative boundaries south", "administrative boundaries east")
    tools = build_tools()
    query_catalogs("Atlantis", BOUNDARIES, store, tools, _embedder())
    assert tools.requests_made == 1


def test_an_exhausted_budget_does_not_kill_the_run(tmp_path):
    store = _store_with(tmp_path, "administrative boundaries, whole country")
    tools = build_tools()
    tools.requests_made = tools.max_requests
    assert query_catalogs("Atlantis", BOUNDARIES, store, tools, _embedder()) == []


def test_hits_are_scoped_to_the_country(tmp_path):
    store = _store_with(tmp_path, "administrative boundaries, whole country")
    assert query_catalogs("Elsewhere", BOUNDARIES, store, build_tools(), _embedder()) == []
