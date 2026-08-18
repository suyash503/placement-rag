"""Create the two Atlas search indexes the retriever depends on.

Hybrid retrieval needs both: a vector index for semantic similarity and a
full-text (BM25) index for lexical matching. Missing the second one is easy to
overlook because vector search keeps working — it just quietly loses every query
that hinges on an exact company name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_mongodb.index import (  # noqa: E402
    create_fulltext_search_index,
    create_vector_search_index,
    update_vector_search_index,
)
from pymongo.errors import OperationFailure  # noqa: E402

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.logging import get_logger  # noqa: E402
from backend.app.rag.documents import FILTERABLE_FIELDS  # noqa: E402
from backend.app.rag.store import get_collection  # noqa: E402

log = get_logger("build_indexes")


def main() -> int:
    settings = get_settings()
    collection = get_collection()
    existing = {idx["name"] for idx in collection.list_search_indexes()}

    verb = "updating" if settings.vector_index_name in existing else "creating"
    build = update_vector_search_index if verb == "updating" else create_vector_search_index

    log.info(
        "%s vector index %r (%d dims, cosine, %d filterable fields)",
        verb, settings.vector_index_name, settings.embedding_dimensions, len(FILTERABLE_FIELDS),
    )
    build(
        collection=collection,
        index_name=settings.vector_index_name,
        path="embedding",
        dimensions=settings.embedding_dimensions,
        similarity="cosine",
        filters=FILTERABLE_FIELDS,
        wait_until_complete=300,
    )
    log.info("  ready")

    if settings.fulltext_index_name in existing:
        log.info("full-text index %r already exists", settings.fulltext_index_name)
    else:
        log.info("creating full-text index %r on 'text'", settings.fulltext_index_name)
        try:
            create_fulltext_search_index(
                collection=collection,
                index_name=settings.fulltext_index_name,
                field="text",
                wait_until_complete=300,
            )
            log.info("  ready")
        except OperationFailure as exc:
            log.error("could not create the full-text index: %s", exc)
            log.error("free-tier clusters allow a limited number of search indexes — drop an unused one and retry")
            return 1

    for idx in collection.list_search_indexes():
        log.info("index %-14s type=%-11s status=%s", idx["name"], idx.get("type", "search"), idx.get("status"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
