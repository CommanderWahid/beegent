"""Embed stored runs and links that have no vector, or one from a different model."""
import sys
from beegent import config
from beegent.embedding import make_embedder
from beegent.store import SqliteStore

BATCH = 256


def _backfill(store: SqliteStore, kind: str, embed) -> None:
    """Both tables move the same way, so they share one loop rather than two copies."""
    pending, text, uid, write = {
        "run": (store.runs_needing_embedding, "use_case", "run_uid", store.set_run_embedding),
        "link": (store.links_needing_embedding, "dataset", "link_uid", store.set_embedding),
    }[kind]

    todo = pending(embed.name)
    print(f"{kind}s to embed: {len(todo)}")
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        for row, vec in zip(chunk, embed([r[text] for r in chunk])):
            write(row[uid], vec, embed.name)
        print(f"  embedded {start + len(chunk)}/{len(todo)}")
    print(f"  {len(pending(embed.name))} {kind}(s) still unembedded")


def main() -> None:
    path = config.BEEGENT_DB
    if not path:
        sys.exit("set BEEGENT_DB to the store you want to backfill")
    embed = make_embedder()
    if embed is None:
        sys.exit("no embedder: install the extra with `uv sync --extra embeddings`")

    store = SqliteStore(path)  # no embedder here: this tool writes the vectors itself
    print(f"model: {embed.name}")
    for kind in ("run", "link"):
        _backfill(store, kind, embed)


if __name__ == "__main__":
    main()
