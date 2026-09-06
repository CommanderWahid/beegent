"""Measure the catalog's similarity threshold against the links actually stored.

    python -m tools.calibrate_relevance ["a use case" ...]

Prints each query's ranked scores, the band the data implies, and the threshold that satisfies
every query. Re-run it whenever EMBED_MODEL changes: cosine values are not comparable across
models, and a guessed threshold silently mis-filters the catalog.
"""
import sys
from beegent import config
from beegent.embedding import band, dot, from_blob, intersect, make_embedder
from beegent.store import SqliteStore

DEFAULT_QUERIES = [
    "administrative boundaries from an OGC API-Features or WFS endpoint",
    "building footprints as a GeoPackage file",
]


def main() -> None:
    queries = sys.argv[1:] or DEFAULT_QUERIES
    if not config.BEEGENT_DB:
        sys.exit("BEEGENT_DB is empty - point it at the store you want to calibrate against")
    embed = make_embedder()
    if embed is None:
        sys.exit("no embedder: install the extra with `uv sync --extra embeddings`")

    links = SqliteStore(config.BEEGENT_DB).embedded_links(embed.name)
    print(f"model: {embed.name}\nlinks: {len(links)}   queries: {len(queries)}\n")
    if len(links) < 2:
        sys.exit("need at least 2 links embedded with this model before a threshold means anything")

    bands = []
    for query, qv in zip(queries, embed(queries)):
        scored = sorted(((dot(qv, from_blob(h["embedding"])), h["dataset"]) for h in links),
                        reverse=True)
        print(query)
        for score, dataset in scored:
            print(f"   {score:.3f}  {dataset[:64]}")
        bands.append(band([s for s, _ in scored]))
        print(f"   -> band {bands[-1][0]:.3f} - {bands[-1][1]:.3f}\n" if bands[-1]
              else "   -> no gap\n")

    agreed = intersect(bands)
    if agreed is None:
        # Averaging would hide that no threshold works, which is the whole point of saying it.
        print("NO single threshold separates every query - the bands do not overlap.")
        print("Either the model is too weak for these datasets, or a query is ambiguous.")
        for query, b in zip(queries, bands):
            print(f"   {b[0]:.3f} - {b[1]:.3f}   {query}" if b else f"   (none)   {query}")
        raise SystemExit(1)

    low, high = agreed
    print(f"agreed band: {low:.3f} - {high:.3f}")
    print(f"suggested CATALOG_MIN_RELEVANCE = {(low + high) / 2:.2f}")


if __name__ == "__main__":
    main()
