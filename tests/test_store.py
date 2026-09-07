#!/usr/bin/env python3
"""Cross-run memory: the SQLite store and its opt-in default."""

import sqlite3

from beegent import config
from beegent.embedding import dot, from_blob, normalise
from beegent.schemas import Candidate, DiscoveryRun
from beegent.store import NullStore, SqliteStore, make_store


def _run(country="Atlantis", status="ok", note=None, decision=None, candidates=()):
    return DiscoveryRun(country=country, use_case="land cover", status=status,
                        critic_decision=decision, critic_note=note,
                        totals={"total_tokens": 4242}, candidates=list(candidates))


def _verified(resource="https://api.atlantis.example/f.parquet"):
    return Candidate(url="https://geo.atlantis.example/p", title="land cover",
                     source="geofetch", confidence=0.95, resource_url=resource,
                     claim={"edition": "2025"}, verification={"ok": True, "status": 206})


def test_persistence_is_opt_in(monkeypatch):
    """Unset means no persistence beyond candidate_list.json - the CLI stays standalone."""
    monkeypatch.setattr("beegent.config.BEEGENT_DB", "")
    assert isinstance(make_store(), NullStore)


def test_a_null_store_answers_everything_emptily():
    s = NullStore()
    s.record_run(_run(), [])
    assert s.prior_runs("Atlantis", 3) == []
    assert s.verified_links("Atlantis") == []


def test_a_missing_parent_directory_is_created(tmp_path):
    """A real run lost its whole store to a folder that did not exist yet."""
    store = SqliteStore(str(tmp_path / "deep" / "nested" / "m.db"))
    store.record_run(_run(), [])
    assert len(store.prior_runs("Atlantis", 3)) == 1


def test_an_unusable_path_degrades_instead_of_failing(tmp_path):
    """A broken store must never cost a run - here the parent is a FILE, so it cannot be made."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    assert isinstance(make_store(str(blocker / "x.db")), NullStore)


def test_a_run_round_trips(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(note="try MLIT instead", decision="replan"),
                     [{"url": "https://dead.example/", "failure_reason": "unreachable"}])
    prior = store.prior_runs("Atlantis", 3)
    assert len(prior) == 1
    assert prior[0]["critic_note"] == "try MLIT instead"
    assert prior[0]["tried"][0]["url"] == "https://dead.example/"


def test_prior_runs_are_newest_first_and_scoped_to_the_country(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))
    for i in range(3):
        store.record_run(_run(note=f"note {i}"), [])
    store.record_run(_run(country="Elsewhere", note="other country"), [])
    prior = store.prior_runs("Atlantis", 2)
    assert [p["critic_note"] for p in prior] == ["note 2", "note 1"]
    assert all(p["critic_note"] != "other country" for p in prior)


def test_verified_candidates_become_links(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    links = store.verified_links("Atlantis")
    assert len(links) == 1
    assert links[0]["resource_url"] == "https://api.atlantis.example/f.parquet"


def test_a_candidate_without_a_resource_url_is_not_a_link(tmp_path):
    """links holds verified endpoints only - that is the whole point of the table."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[Candidate(url="u", title="t", source="geofetch")]), [])
    assert store.verified_links("Atlantis") == []


def test_the_same_resource_url_is_stored_once(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    store.record_run(_run(candidates=[_verified()]), [])
    assert len(store.verified_links("Atlantis")) == 1


def test_query_parameters_are_distinct_resources(tmp_path):
    """UNIQUE is on the RAW url: normalize_url() strips the query and would collapse these."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[
        _verified("https://x.example/api?identifier=N03"),
        _verified("https://x.example/api?identifier=N04")]), [])
    assert len(store.verified_links("Atlantis")) == 2


def test_a_link_carries_the_run_that_found_it(tmp_path):
    """run_uid is why first_seen needs no column of its own."""
    path = str(tmp_path / "m.db")
    SqliteStore(path).record_run(_run(candidates=[_verified()]), [])
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT l.run_uid, r.run_uid FROM links l "
                          "JOIN runs r ON r.run_uid = l.run_uid").fetchall()
    assert len(rows) == 1 and rows[0][0] == rows[0][1]


# --- links is a LOG: the same URL may be several datasets, and a re-probe must land ---


def test_re_verifying_a_url_appends_and_the_newest_wins(tmp_path):
    """UNIQUE + INSERT OR IGNORE dropped the newer row, so a just-proven link looked stale."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    store.record_run(_run(candidates=[_verified()]), [])
    links = store.verified_links("Atlantis")
    assert len(links) == 1, "one URL, one row returned"
    with sqlite3.connect(str(tmp_path / "m.db")) as db:
        stamps = [r[0] for r in db.execute("SELECT last_verified FROM links")]
    assert len(stamps) == 2, "both observations kept - links is a log"
    assert links[0]["last_verified"] == max(stamps), "the newest verification is the one served"


def test_one_file_can_serve_two_datasets(tmp_path):
    """A GeoPackage holds several layers; UNIQUE let only the first use case store it."""
    store = SqliteStore(str(tmp_path / "m.db"))
    shared = "https://api.atlantis.example/all_boundaries.gpkg"
    store.record_run(_run(candidates=[_verified(shared)]), [])
    store.record_run(_run(candidates=[_verified(shared)]), [])
    assert len(store.verified_links("Atlantis")) == 1, "deduped to the newest observation"


def test_marking_the_newest_dead_does_not_resurface_an_older_ok(tmp_path):
    """Filtering status BEFORE deduping would hand back a stale row as if it were live."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    store.record_run(_run(candidates=[_verified()]), [])
    store.mark_link(store.verified_links("Atlantis")[0]["link_uid"], "ko")
    assert store.verified_links("Atlantis") == []


def test_the_dataset_column_holds_the_dataset_not_the_description(tmp_path):
    """`title` is the angle DESCRIPTION - it names a publisher and a channel, which is noise."""
    store = SqliteStore(str(tmp_path / "m.db"))
    cand = _verified()
    cand.title = "Fetch the Kadaster/PDOK OGC API Features endpoint for boundaries"
    cand.dataset = "administrative boundaries, whole country"
    store.record_run(_run(candidates=[cand]), [])
    assert store.verified_links("Atlantis")[0]["dataset"] == cand.dataset


# --- embeddings: written when available, absent when not, never fatal ---


def _fake_embed(name="fake-model"):
    """Deterministic unit vectors, so the suite needs no model download."""
    def embed(texts):
        return [normalise([float(len(t)), 1.0, 0.5]) for t in texts]
    embed.name = name
    return embed


def test_a_vector_and_its_model_name_are_stored_together(tmp_path):
    """A vector is only comparable within one model, so the name must ride along."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_fake_embed())
    store.record_run(_run(candidates=[_verified()]), [])
    with sqlite3.connect(str(tmp_path / "m.db")) as db:
        blob, model = db.execute("SELECT embedding, embed_model FROM links").fetchone()
    assert model == "fake-model"
    assert abs(dot(from_blob(blob), from_blob(blob)) - 1.0) < 1e-5, "stored unit length"


def test_no_embedder_means_null_vectors_not_an_error(tmp_path):
    """The default, and what a machine without the extra installed does."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    with sqlite3.connect(str(tmp_path / "m.db")) as db:
        assert db.execute("SELECT embedding, embed_model FROM links").fetchone() == (None, None)
    assert len(store.verified_links("Atlantis")) == 1, "the link is still stored"


def test_a_failing_embedder_costs_the_vector_not_the_link(tmp_path):
    """Losing a verified download because a model was unavailable is the wrong trade."""
    def boom(texts):
        raise RuntimeError("model gone")
    boom.name = "broken"

    store = SqliteStore(str(tmp_path / "m.db"), embed=boom)
    store.record_run(_run(candidates=[_verified()]), [])
    assert len(store.verified_links("Atlantis")) == 1
    with sqlite3.connect(str(tmp_path / "m.db")) as db:
        assert db.execute("SELECT embedding FROM links").fetchone()[0] is None


def test_a_run_embeds_its_links_in_one_batched_call(tmp_path):
    """Batching matters for a backfill; a run just must not call once per link."""
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return [normalise([1.0, 0.0]) for _ in texts]
    counting.name = "counting"

    store = SqliteStore(str(tmp_path / "m.db"), embed=counting)
    store.record_run(_run(candidates=[_verified("https://a.example/1.parquet"),
                                      _verified("https://a.example/2.parquet")]), [])
    assert calls == [3], "one call: the run's use case, then both links"


def test_the_embedded_text_is_the_dataset_not_the_description(tmp_path):
    """Whatever is embedded is what the catalog matches on."""
    seen = []

    def capture(texts):
        seen.extend(texts)
        return [normalise([1.0, 0.0]) for _ in texts]
    capture.name = "capture"

    cand = _verified()
    cand.title = "Fetch the Kadaster/PDOK OGC API Features endpoint"
    cand.dataset = "administrative boundaries, whole country"
    SqliteStore(str(tmp_path / "m.db"), embed=capture).record_run(_run(candidates=[cand]), [])
    # The run's use case leads the batch; the LINK's text is the dataset, never the description.
    assert seen == ["land cover", "administrative boundaries, whole country"]


# --- backfill: a model change invalidates every stored vector at once ---


def test_links_with_no_vector_need_embedding(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))  # no embedder -> NULL vectors
    store.record_run(_run(candidates=[_verified()]), [])
    assert len(store.links_needing_embedding("any-model")) == 1


def test_a_model_change_invalidates_every_stored_vector(tmp_path):
    """The realistic trigger: not missing vectors, but EMBED_MODEL moving under them."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_fake_embed("model-a"))
    store.record_run(_run(candidates=[_verified()]), [])
    assert store.links_needing_embedding("model-a") == [], "current model: nothing to do"
    assert len(store.links_needing_embedding("model-b")) == 1, "another model: all stale"


def test_setting_an_embedding_makes_the_link_matchable(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    todo = store.links_needing_embedding("model-a")
    store.set_embedding(todo[0]["link_uid"], normalise([1.0, 0.0]), "model-a")
    assert store.links_needing_embedding("model-a") == []
    link = store.verified_links("Atlantis")[0]
    assert link["embed_model"] == "model-a"
    assert from_blob(link["embedding"]) == [1.0, 0.0]


def test_backfill_is_idempotent(tmp_path):
    """Safe to re-run: the second pass finds nothing, which is how you verify it worked."""
    store = SqliteStore(str(tmp_path / "m.db"))
    store.record_run(_run(candidates=[_verified()]), [])
    for row in store.links_needing_embedding("m"):
        store.set_embedding(row["link_uid"], normalise([1.0, 0.0]), "m")
    assert store.links_needing_embedding("m") == []


def test_memory_is_on_by_default(monkeypatch, tmp_path):
    """No env var needed: an unset BEEGENT_DB now means a real store, not a NullStore."""
    monkeypatch.setattr(config, "BEEGENT_DB", str(tmp_path / "auto" / "memory.db"))
    assert isinstance(make_store(), SqliteStore)


def test_an_empty_path_is_the_off_switch(monkeypatch):
    monkeypatch.setattr(config, "BEEGENT_DB", "")
    assert isinstance(make_store(), NullStore)


def test_embedded_links_are_filtered_by_model_in_sql(tmp_path):
    """Comparing across vector spaces is silent nonsense, so the filter is not optional."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_fake_embed("model-a"))
    store.record_run(_run(candidates=[_verified()]), [])
    assert len(store.embedded_links("model-a")) == 1
    assert store.embedded_links("model-b") == []


def test_a_link_with_no_vector_is_not_an_embedded_link(tmp_path):
    store = SqliteStore(str(tmp_path / "m.db"))  # no embedder -> NULL vector
    store.record_run(_run(candidates=[_verified()]), [])
    assert store.embedded_links("any") == []


# --- run memory is narrowed by USE CASE, not just by country ---


def _embedder(mapping, name="fake"):
    """Deterministic unit vectors: same key -> cosine 1.0, different key -> 0.0."""
    def embed(texts):
        return [normalise(mapping[t]) for t in texts]
    embed.name = name
    return embed


AXES = {"land cover": [1.0, 0.0], "building footprints": [0.0, 1.0]}


def test_a_run_about_another_use_case_is_not_replayed(tmp_path):
    """A boundaries run fed dead URLs from a footprints run is noise, not memory."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_embedder(AXES))
    store.record_run(_run(), [{"url": "https://cover.example/", "failure_reason": "x"}])
    store.record_run(DiscoveryRun(country="Atlantis", use_case="building footprints",
                                  status="ok", totals={"total_tokens": 1}),
                     [{"url": "https://footprints.example/", "failure_reason": "y"}])

    prior = store.prior_runs("Atlantis", 3, "land cover")
    assert [r["use_case"] for r in prior] == ["land cover"]
    assert prior[0]["tried"][0]["url"] == "https://cover.example/"


def test_matching_runs_are_still_newest_first(tmp_path):
    """Relevance FILTERS; recency still ORDERS."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_embedder(AXES))
    store.record_run(_run(note="older"), [])
    store.record_run(_run(country="Elsewhere", note="another country"), [])
    store.record_run(_run(note="newer"), [])

    prior = store.prior_runs("Atlantis", 3, "land cover")
    assert [r["critic_note"] for r in prior] == ["newer", "older"]


def test_the_cap_is_applied_after_the_filter(tmp_path):
    """Filtering after the LIMIT would let irrelevant rows crowd out the relevant ones."""
    store = SqliteStore(str(tmp_path / "m.db"), embed=_embedder(AXES))
    for _ in range(4):  # four newer, irrelevant runs
        store.record_run(DiscoveryRun(country="Atlantis", use_case="building footprints",
                                      status="ok", totals={}), [])
    store.record_run(_run(note="the one that matters"), [])

    prior = store.prior_runs("Atlantis", 2, "land cover")
    assert [r["critic_note"] for r in prior] == ["the one that matters"]


def test_with_no_embedder_prior_runs_falls_back_to_pure_recency(tmp_path):
    """Embeddings are optional; their absence must never drop a run."""
    store = SqliteStore(str(tmp_path / "m.db"))  # no embedder at all
    store.record_run(_run(note="a"), [])
    store.record_run(DiscoveryRun(country="Atlantis", use_case="something else",
                                  status="ok", critic_note="b", totals={}), [])

    assert len(store.prior_runs("Atlantis", 3, "land cover")) == 2, "nothing filtered"


def test_a_run_embedded_by_another_model_is_not_compared(tmp_path):
    """A vector from another model is not comparable, and comparing anyway is nonsense."""
    path = str(tmp_path / "m.db")
    SqliteStore(path, embed=_embedder(AXES, name="old")).record_run(_run(), [])

    store = SqliteStore(path, embed=_embedder(AXES, name="new"))
    assert store.prior_runs("Atlantis", 3, "land cover") == [], "excluded, not scored"
    assert [r["use_case"] for r in store.runs_needing_embedding("new")] == ["land cover"]


def test_a_backfilled_run_becomes_matchable_again(tmp_path):
    """This is the recovery path for an EMBED_MODEL change."""
    path = str(tmp_path / "m.db")
    SqliteStore(path, embed=_embedder(AXES, name="old")).record_run(_run(), [])

    embed = _embedder(AXES, name="new")
    store = SqliteStore(path, embed=embed)
    for row in store.runs_needing_embedding("new"):
        store.set_run_embedding(row["run_uid"], embed([row["use_case"]])[0], "new")

    assert len(store.prior_runs("Atlantis", 3, "land cover")) == 1
    assert store.runs_needing_embedding("new") == [], "re-running is idempotent"


# --- a model change repairs itself at the start of the next run ---


def test_a_model_change_re_embeds_both_tables(tmp_path):
    """Silence is the symptom of a model change, so the run must not depend on remembering a tool."""
    path = str(tmp_path / "m.db")
    old = _embedder(AXES, name="old")
    SqliteStore(path, embed=old).record_run(_run(candidates=[_verified()]), [])

    store = SqliteStore(path, embed=_embedder(AXES | {"land cover, whole country": [1.0, 0.0]},
                                              name="new"))
    assert store.prior_runs("Atlantis", 3, "land cover") == [], "unmatchable before the sync"
    assert store.sync_embeddings() == {"runs": 1, "links": 1}
    assert len(store.prior_runs("Atlantis", 3, "land cover")) == 1, "matchable again"
    assert store.sync_embeddings() == {}, "and re-running finds nothing left to do"


def test_an_unchanged_model_costs_no_embedding_call(tmp_path):
    """The common path is 'nothing changed', so it must not re-embed the whole store every run."""
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return [normalise([1.0, 0.0]) for _ in texts]
    counting.name = "same"

    store = SqliteStore(str(tmp_path / "m.db"), embed=counting)
    store.record_run(_run(candidates=[_verified()]), [])
    calls.clear()

    assert store.sync_embeddings() == {}
    assert calls == [], "no work means no model call at all"


def test_with_no_embedder_sync_leaves_every_vector_untouched(tmp_path):
    """No model is not a reason to discard what is stored - keep the state as it is."""
    path = str(tmp_path / "m.db")
    SqliteStore(path, embed=_embedder(AXES, name="old")).record_run(_run(), [])

    assert SqliteStore(path).sync_embeddings() == {}, "no embedder, no work"
    after = SqliteStore(path, embed=_embedder(AXES, name="old"))
    assert len(after.prior_runs("Atlantis", 3, "land cover")) == 1, "the old vector survived"
