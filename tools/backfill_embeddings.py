"""Embed stored links that have no vector, or one from a different model."""
import sys
from beegent import config
from beegent.embedding import make_embedder
from beegent.store import SqliteStore

BATCH = 256


def main() -> None:
    path = config.BEEGENT_DB
    if not path:
        sys.exit("set BEEGENT_DB to the store you want to backfill")
    embed = make_embedder()
    if embed is None:
        sys.exit("no embedder: install the extra with `uv sync --extra embeddings`")

    store = SqliteStore(path)  # no embedder here: this tool writes the vectors itself
    todo = store.links_needing_embedding(embed.name)
    print(f"model: {embed.name}\nlinks to embed: {len(todo)}")
    if not todo:
        return

    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        vectors = embed([r["dataset"] for r in chunk])
        for row, vec in zip(chunk, vectors):
            store.set_embedding(row["link_uid"], vec, embed.name)
        print(f"  embedded {start + len(chunk)}/{len(todo)}")

    left = len(store.links_needing_embedding(embed.name))
    print(f"done - {left} link(s) still unembedded")


if __name__ == "__main__":
    main()
