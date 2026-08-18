import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.logging import get_logger  # noqa: E402
from backend.app.rag.documents import build_documents, load_rows  # noqa: E402
from backend.app.rag.store import get_collection, get_vector_store  # noqa: E402

log = get_logger("rag.ingest")


def ingest(batch_size: int = 200, drop: bool = False) -> int:
    settings = get_settings()
    csv_path = settings.enriched_csv

    if not csv_path.exists():
        log.error("%s not found — run backend/scripts/enrich_dataset.py first", csv_path)
        return 1

    rows = load_rows(csv_path)
    documents = build_documents(rows)
    ids = [d.metadata["doc_id"] for d in documents]

    record_count = sum(1 for d in documents if d.metadata["doc_type"] == "record")
    log.info(
        "built %d documents (%d record, %d company) from %d rows",
        len(documents), record_count, len(documents) - record_count, len(rows),
    )

    collection = get_collection()
    if drop:
        log.warning("dropping existing collection %s", collection.name)
        collection.delete_many({})

    before = collection.count_documents({})
    store = get_vector_store()

    started = time.perf_counter()
    for start in range(0, len(documents), batch_size):
        chunk = documents[start : start + batch_size]
        store.add_documents(chunk, ids=ids[start : start + batch_size])
        log.info("  upserted %d/%d", min(start + batch_size, len(documents)), len(documents))

    after = collection.count_documents({})
    log.info(
        "done in %.1fs — collection went %d -> %d documents",
        time.perf_counter() - started, before, after,
    )

    if before and after == before:
        log.info("count unchanged, so the re-ingest updated in place rather than duplicating")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed placement records into MongoDB Atlas")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--drop", action="store_true", help="clear the collection first")
    args = parser.parse_args()
    return ingest(batch_size=args.batch_size, drop=args.drop)


if __name__ == "__main__":
    raise SystemExit(main())
